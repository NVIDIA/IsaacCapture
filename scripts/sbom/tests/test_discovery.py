# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build provenance discovery tests."""

from __future__ import annotations

import json
import os
import subprocess
import zipfile
import zlib
from pathlib import Path

import pytest
import synth
from isaaccapture_sbom import discovery


def test_source_trees_are_found_without_being_listed(workspace):
    components = discovery.discover_source_trees(workspace.build / "_deps")

    assert set(components) == {
        "alpha",
        "beta",
        "delta",
        "gamma",
        "license-list-data",
        "omega",
        "vendorpy",
    }


def test_identity_comes_off_the_remote_url(workspace):
    alpha = discovery.discover_source_trees(workspace.build / "_deps")["alpha"]

    assert alpha.supplier == "Organization: example-org"
    assert alpha.homepage == "https://github.com/example-org/alpha"
    # With no checkout, the declared tag stands in for a commit throughout.
    assert alpha.purl == "pkg:github/example-org/alpha@v1.2.3"
    assert (
        alpha.download_location == "git+https://github.com/example-org/alpha.git@v1.2.3"
    )
    assert alpha.version == "v1.2.3"


SDL_RELEASE = (
    "https://github.com/libsdl-org/SDL/releases/download/"
    "release-2.32.10/SDL2-2.32.10.tar.gz"
)
SDL_HASH = "SHA256=" + "5f" * 32


@pytest.mark.parametrize("hook", [False, True], ids=["shell", "under-a-hook"])
def test_an_archive_fetched_inside_a_checkout_is_not_that_checkout(
    workspace, monkeypatch, hook
):
    """GIT_DIR can override -C, even for an archive nested under the project checkout."""
    deps = workspace.build / "_deps"
    synth.write(deps / "sdl2-src/LICENSE.txt", synth.spdx_text("Zlib"))
    synth.declare(deps, "sdl2", urls=(SDL_RELEASE,), digest=SDL_HASH)
    synth.git(
        workspace.root,
        "remote",
        "add",
        "origin",
        "https://github.com/example/enclosing-project.git",
    )
    if hook:
        monkeypatch.setenv("GIT_DIR", str(workspace.root / ".git"))

    components = discovery.discover_source_trees(deps)
    sdl2 = components["sdl2"]

    assert (sdl2.name, sdl2.supplier, sdl2.homepage, sdl2.purl, sdl2.version) == (
        "SDL",
        "Organization: libsdl-org",
        "https://github.com/libsdl-org/SDL",
        "pkg:github/libsdl-org/sdl@release-2.32.10",
        "release-2.32.10",
    )
    assert sdl2.download_location == SDL_RELEASE
    assert f"Declared archive: {SDL_RELEASE} ({SDL_HASH})." in sdl2.source_info
    # A tree declared by GIT_REPOSITORY with no .git left takes no describe either.
    for component in components.values():
        assert "enclosing-project" not in repr(component)
        assert "git describe" not in component.source_info


def test_the_checkout_itself_is_read_whatever_git_dir_says(
    workspace, tmp_path, monkeypatch
):
    head = discovery.git(workspace.root, "rev-parse", "HEAD")
    synth.clone_of(tmp_path / "other", "https://example.invalid/other.git", "v0")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "other" / ".git"))

    assert discovery.git(workspace.root, "rev-parse", "HEAD") == head
    assert "VERSION" in discovery.tracked_files(workspace.root)


DIRECT = pytest.mark.parametrize("direct", [False, True], ids=["sub-build", "direct"])


@DIRECT
@pytest.mark.parametrize(
    ("urls", "digest", "purl", "version", "downloadable"),
    [
        (
            ("https://downloads.example.org/foo-1.4.2.tar.gz",),
            "SHA256=AB12",
            "pkg:generic/foo@1.4.2?checksum=sha256:ab12"
            "&download_url=https://downloads.example.org/foo-1.4.2.tar.gz",
            "1.4.2",
            True,
        ),
        (
            (
                "https://mirror-a.example.org/foo-1.4.2.tar.gz",
                "https://mirror-b.example.org/foo-1.4.3.tar.gz",
            ),
            "SHA3_256=CD34",
            "pkg:generic/foo@1.4.2?checksum=sha3-256:cd34"
            "&download_url=https://mirror-a.example.org/foo-1.4.2.tar.gz",
            "1.4.2",
            True,
        ),
        (
            ("https://gitlab.com/example-org/foo/-/archive/v1.5.0/foo-v1.5.0.tar.gz",),
            None,
            "pkg:generic/foo@1.5.0?download_url="
            "https://gitlab.com/example-org/foo/-/archive/v1.5.0/foo-v1.5.0.tar.gz",
            "1.5.0",
            True,
        ),
        (
            ("https://downloads.sourceforge.net/project/foo/foo-2.3.tar.bz2",),
            None,
            "pkg:generic/foo@2.3"
            "?download_url=https://downloads.sourceforge.net/project/foo/foo-2.3.tar.bz2",
            "2.3",
            True,
        ),
        (
            ("/home/builder/archives/foo-1.4.2.tar.gz",),
            None,
            "pkg:generic/foo@1.4.2",
            "1.4.2",
            False,
        ),
        (
            ("file:///home/builder/archives/foo.zip",),
            None,
            "pkg:generic/foo",
            "NOASSERTION",
            False,
        ),
        (
            ("https://github.com/example-org/foo/archive/refs/tags/v3.0.tar.gz",),
            None,
            "pkg:github/example-org/foo@v3.0",
            "v3.0",
            True,
        ),
        (
            ("https://github.com/example-org/foo/archive/0a1b2c3d.zip",),
            None,
            "pkg:github/example-org/foo@0a1b2c3d",
            "0a1b2c3d",
            True,
        ),
        (
            (
                "https://github.com/example-org/foo/releases/download/v1.0/foo-linux.zip",
            ),
            None,
            "pkg:github/example-org/foo@v1.0",
            "v1.0",
            True,
        ),
        (
            (
                "https://github.com/example-org/foo/archive/refs/tags/"
                "releases/cpp/v2.1.1.tar.gz",
            ),
            None,
            "pkg:github/example-org/foo@releases/cpp/v2.1.1",
            "releases/cpp/v2.1.1",
            True,
        ),
        (
            ("https://codeload.github.com/example-org/foo/tar.gz/refs/tags/v4.2",),
            None,
            "pkg:github/example-org/foo@v4.2",
            "v4.2",
            True,
        ),
    ],
)
def test_a_declared_archive_is_named_by_its_url(
    tmp_path, urls, digest, purl, version, downloadable, direct
):
    deps = tmp_path / "build" / "_deps"
    (deps / "foo-src").mkdir(parents=True)
    synth.declare(deps, "foo", urls=urls, digest=digest, direct=direct)

    foo = discovery.discover_source_trees(deps)["foo"]

    download = urls[0] if downloadable else "NOASSERTION"
    assert (foo.purl, foo.version, foo.download_location) == (purl, version, download)
    github = purl.startswith("pkg:github/")
    assert foo.supplier == ("Organization: example-org" if github else "NOASSERTION")
    assert "/home/builder" not in foo.source_info


@DIRECT
def test_a_declaration_is_matched_to_the_tree_it_names(tmp_path, direct):
    """`license_list_data` fills `license-list-data-src`: the content name is not the tree."""
    deps = tmp_path / "build" / "_deps"
    (deps / "foo-bar-src").mkdir(parents=True)
    synth.declare(
        deps,
        "foo_bar",
        repository="https://github.com/example-org/foo-bar.git",
        tag="v5",
        source="foo-bar-src",
        direct=direct,
    )

    foo_bar = discovery.discover_source_trees(deps)["foo-bar"]

    assert (foo_bar.purl, foo_bar.version) == (
        "pkg:github/example-org/foo-bar@v5",
        "v5",
    )
    assert "Declared constraint: v5." in foo_bar.source_info


@pytest.mark.parametrize("stale_is_newer", [False, True], ids=["older", "newer"])
def test_a_stamp_left_by_an_earlier_declaration_does_not_count(
    tmp_path, stale_is_newer
):
    """Switching method leaves the other stamp behind, and CMake rewrites a
    stamp only when its contents change, so its mtime cannot decide."""
    deps = tmp_path / "build" / "_deps"
    (deps / "foo-src").mkdir(parents=True)
    synth.declare(deps, "foo", repository="https://example.invalid/foo.git", tag="v1")
    synth.declare(deps, "foo", urls=("https://downloads.example.org/foo-2.0.tar.gz",))
    git_stamp = next(deps.glob("foo-subbuild/**/*-gitinfo.txt"))
    url_stamp = next(deps.glob("foo-subbuild/**/*-urlinfo.txt"))
    stale, current = (git_stamp, url_stamp)
    os.utime(stale, ns=(2, 2) if stale_is_newer else (0, 0))
    os.utime(current, ns=(1, 1))

    foo = discovery.discover_source_trees(deps)["foo"]

    assert (foo.version, foo.purl.split("?")[0]) == ("2.0", "pkg:generic/foo@2.0")
    assert "example.invalid" not in repr(foo)


def test_after_a_policy_change_the_newer_layout_is_the_declaration(tmp_path):
    """Sub-build stamps stay behind when CMP0168 turns NEW, and the reverse."""
    deps = tmp_path / "build" / "_deps"
    (deps / "foo-src").mkdir(parents=True)
    synth.declare(deps, "foo", urls=("https://downloads.example.org/foo-1.0.tar.gz",))
    synth.declare(
        deps, "foo", urls=("https://downloads.example.org/foo-2.0.tar.gz",), direct=True
    )
    os.utime(next(deps.glob("foo-subbuild/**/*-urlinfo.txt")), ns=(0, 0))

    assert discovery.discover_source_trees(deps)["foo"].version == "2.0"


def test_a_checkout_takes_its_git_stamp_over_a_newer_url_stamp(tmp_path):
    deps = tmp_path / "build" / "_deps"
    commit = synth.clone_of(deps / "foo-src", "https://git.example.org/foo.git", "v3")
    synth.declare(deps, "foo", repository="https://git.example.org/foo.git", tag="v3")
    synth.declare(deps, "foo", urls=("https://downloads.example.org/foo-2.0.tar.gz",))
    os.utime(next(deps.glob("foo-subbuild/**/*-gitinfo.txt")), ns=(0, 0))

    foo = discovery.discover_source_trees(deps)["foo"]

    assert foo.version == commit
    assert "Declared constraint: v3." in foo.source_info
    assert "downloads.example.org" not in repr(foo)


def test_git_config_in_the_environment_does_not_choose_the_remote(
    tmp_path, monkeypatch
):
    deps = tmp_path / "build" / "_deps"
    synth.clone_of(deps / "foo-src", "https://git.example.org/foo.git", "v1")
    synth.clone_of(tmp_path / "other", "https://git.example.org/other.git", "v1")
    monkeypatch.setenv("GIT_CONFIG", str(tmp_path / "other" / ".git" / "config"))

    foo = discovery.discover_source_trees(deps)["foo"]

    assert foo.homepage == "https://git.example.org/foo.git"


@pytest.mark.parametrize(
    ("repository", "tag", "purl", "download"),
    [
        (
            "https://git.example.org/foo.git",
            "v1",
            "pkg:generic/foo@v1",
            "git+https://git.example.org/foo.git@v1",
        ),
        (
            "https://git.example.org/foo.git",
            "v1.0+build",
            "pkg:generic/foo@v1.0%2Bbuild",
            "git+https://git.example.org/foo.git@v1.0+build",
        ),
        (
            "git@git.example.org:org/foo.git",
            "v1",
            "pkg:generic/foo@v1",
            "git+ssh://git@git.example.org/org/foo.git@v1",
        ),
        ("/srv/git/foo.git", "v1", "pkg:generic/foo@v1", "NOASSERTION"),
    ],
)
def test_a_declared_repository_off_github_is_named_by_its_tag(
    tmp_path, repository, tag, purl, download
):
    deps = tmp_path / "build" / "_deps"
    (deps / "foo-src").mkdir(parents=True)
    synth.declare(deps, "foo", repository=repository, tag=tag)

    foo = discovery.discover_source_trees(deps)["foo"]

    assert (foo.purl, foo.download_location, foo.version) == (purl, download, tag)


def test_a_moved_build_tree_still_reads_its_clone_script(tmp_path):
    deps = tmp_path / "old" / "_deps"
    (deps / "foo-src").mkdir(parents=True)
    synth.declare(deps, "foo", repository="https://example.invalid/foo.git", tag="v7")
    (tmp_path / "old").rename(tmp_path / "new")

    foo = discovery.discover_source_trees(tmp_path / "new" / "_deps")["foo"]

    assert foo.version == "v7"


def test_credentials_in_a_declared_url_are_not_published(tmp_path):
    deps = tmp_path / "build" / "_deps"
    (deps / "foo-src").mkdir(parents=True)
    (deps / "bar-src").mkdir(parents=True)
    synth.declare(
        deps,
        "foo",
        urls=("https://u:s3cret@dl.example.org/foo-1.0.tgz?token=t0ken#sig=s1g",),
    )
    synth.declare(deps, "bar", repository="https://tok3n@git.example.org/bar.git")

    components = discovery.discover_source_trees(deps)

    for secret in ("s3cret", "t0ken", "s1g"):
        assert secret not in repr(components["foo"])
    assert "tok3n" not in repr(components["bar"])
    assert components["foo"].download_location == "https://dl.example.org/foo-1.0.tgz"
    assert components["foo"].purl.endswith(
        "download_url=https://dl.example.org/foo-1.0.tgz"
    )


def test_configuration_from_the_environment_reaches_git(workspace, monkeypatch):
    """A container's safe.directory arrives this way; it does not pick a repository."""
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "sbom.probe")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "kept")

    assert discovery.git(workspace.root, "config", "--get", "sbom.probe") == "kept"


@pytest.mark.parametrize(
    ("in_another_checkout", "separate_git_dir", "through_symlink"),
    [(False, False, False), (True, True, False), (False, False, True)],
    ids=["standalone", "submodule-like", "symlinked"],
)
def test_a_checkout_of_its_own_answers_for_itself(
    workspace, tmp_path, in_another_checkout, separate_git_dir, through_symlink
):
    build = workspace.build if in_another_checkout else tmp_path / "build"
    root = build / "_deps" / "epsilon-src"
    synth.write(root / "LICENSE", synth.spdx_text("MIT"))
    commit = synth.clone_of(
        root,
        "https://github.com/example-org/epsilon.git",
        "v2.0.0",
        git_dir=tmp_path / "modules" / "epsilon" if separate_git_dir else None,
    )
    synth.declare(
        build / "_deps", "epsilon", repository="https://example.invalid/x.git"
    )
    if through_symlink:
        (tmp_path / "link").symlink_to(build, target_is_directory=True)
        build = tmp_path / "link"

    epsilon = discovery.discover_source_trees(build / "_deps")["epsilon"]

    assert epsilon.version == commit
    assert epsilon.purl == f"pkg:github/example-org/epsilon@{commit}"
    assert epsilon.download_location.endswith(f"/epsilon.git@{commit}")
    assert "git describe: v2.0.0." in epsilon.source_info


def test_licenses_come_from_the_trees_own_files(workspace):
    components = discovery.discover_source_trees(workspace.build / "_deps")

    assert components["alpha"].license_concluded == "MIT"
    assert components["delta"].license_concluded == "BSD-3-Clause"
    assert components["gamma"].license_concluded == "Zlib"


def test_link_graph_finds_static_and_header_only_contributions(workspace):
    graph = discovery.BuildGraph(workspace.build)
    extension = graph.by_name("_ext.cpython-312-x86_64-linux-gnu.so")

    assert len(extension) == 1
    contributions = graph.contributions(extension[0])

    assert contributions["alpha"] == {"compiled-source", "header-include"}
    # beta ships headers only; an include path is the only trace it leaves.
    assert contributions["beta"] == {"header-include"}


def test_link_graph_recurses_through_static_archives(workspace):
    """delta reaches the wheel only through libdelta.a inside libgamma_renamed.so."""
    graph = discovery.BuildGraph(workspace.build)
    gamma = graph.by_name("libgamma_renamed.so")[0]

    contributions = graph.contributions(gamma)

    assert "gamma" in contributions
    assert "delta" in contributions


def test_first_party_is_decided_by_whose_sources_were_compiled(workspace):
    graph = discovery.BuildGraph(workspace.build)
    repo = workspace.root.resolve()

    extension = graph.by_name("_ext.cpython-312-x86_64-linux-gnu.so")[0]
    gamma = graph.by_name("libgamma_renamed.so")[0]

    assert graph.first_party(extension, repo)
    assert not graph.first_party(gamma, repo)


def test_a_test_only_dependency_reaches_no_shipped_artifact(workspace):
    graph = discovery.BuildGraph(workspace.build)
    extension = graph.by_name("_ext.cpython-312-x86_64-linux-gnu.so")[0]
    gamma = graph.by_name("libgamma_renamed.so")[0]

    assert "omega" not in graph.contributions(extension)
    assert "omega" not in graph.contributions(gamma)
    assert "omega" in graph.contributions(graph.by_name("unit_tests")[0])


def test_archive_members_are_indexed_by_content(workspace):
    index = discovery.discover_archives(
        workspace.root, frozenset(discovery.tracked_files(workspace.root))
    )

    assert synth.SDK_ARCHIVE in index.archives
    demo = index.by_name["libdemo.so"]
    assert len(demo) == 1
    assert index.by_hash[demo[0].sha256].container == synth.SDK_ARCHIVE


def test_archive_version_and_license_come_from_the_archive(workspace):
    index = discovery.discover_archives(
        workspace.root, frozenset(discovery.tracked_files(workspace.root))
    )
    record = index.archives[synth.SDK_ARCHIVE]

    assert record["version_text"] == "1.2.0"
    assert [member.path for member, _ in record["license_members"]] == ["LICENSE.txt"]


def test_an_archive_root_sidecar_is_not_a_license_member(tmp_path):
    archive = tmp_path / "sdk.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("LICENSE.txt", synth.spdx_text("MIT"))
        handle.writestr("libdemo.so", b"\0")
        handle.writestr("libdemo.so.license", "SPDX-License-Identifier: MIT\n")
        handle.writestr("stb.license", synth.spdx_text("MIT"))
    index = discovery.ArchiveIndex()

    index.add_archive(archive, "sdk.zip")

    assert [
        member.path for member, _ in index.archives["sdk.zip"]["license_members"]
    ] == ["LICENSE.txt", "stb.license"]


def test_a_toolkit_licence_beside_a_library_is_not_a_sidecar(tmp_path):
    synth.write(tmp_path / "lib/libx.so.1", "\0")
    synth.write(tmp_path / "lib/libx.so.1.license", "SPDX-License-Identifier: MIT\n")
    synth.write(tmp_path / "EULA.txt", "terms\n")

    assert discovery._license_near(tmp_path / "lib/libx.so.1") == tmp_path / "EULA.txt"


def test_a_build_tree_without_a_file_api_reply_says_so(workspace):
    import shutil

    shutil.rmtree(workspace.build / ".cmake")

    with pytest.raises(discovery.FileApiError, match="cmake_file_api"):
        discovery.BuildGraph(workspace.build, synth.CONFIG)


def test_a_stale_codemodel_beside_the_current_one_is_not_read(workspace):
    """The stale codemodel sorts later; only the index selects the current one."""
    reply = workspace.build / ".cmake" / "api" / "v1" / "reply"
    stale = json.loads((reply / "codemodel-v2-fixture.json").read_text())
    stale["configurations"] = [{"name": synth.CONFIG, "targets": []}]
    (reply / "codemodel-v2-stale.json").write_text(json.dumps(stale))

    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)

    assert graph.by_name("_ext.cpython-312-x86_64-linux-gnu.so")


def test_a_configuration_the_codemodel_does_not_describe_is_refused(workspace):
    with pytest.raises(discovery.FileApiError, match="Debug"):
        discovery.BuildGraph(workspace.build, "Debug")


def test_the_graph_is_read_from_the_codemodel_not_from_generator_output(workspace):
    assert not list(workspace.build.rglob("link.txt"))
    assert not (workspace.build / "compile_commands.json").exists()

    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)

    assert graph.by_name("_ext.cpython-312-x86_64-linux-gnu.so")
    assert graph.by_name("libgamma_renamed.so")


def test_third_party_checked_in_here_is_still_third_party(workspace):
    """Copyright metadata, not directory placement, identifies vendored code."""
    from isaaccapture_sbom import evidence as evidence_module

    space = evidence_module.discover(workspace.root, workspace.build)
    vendored = {
        key: component
        for key, component in space.components.items()
        if component.kind == "vendored-source"
    }

    assert [c.name for c in vendored.values()] == [
        "Vendored source (Upstream Widgets Ltd)"
    ]
    assert [c.supplier for c in vendored.values()] == [
        "Organization: Upstream Widgets Ltd"
    ]
    component = next(iter(vendored.values()))
    assert component.license_concluded == "BSL-1.0"
    # Use upstream reference texts, not the repository's own license pool.
    assert [item.path for item in component.evidence] == ["BSL-1.0.txt"]
    assert all(item.origin == "spdx-reference" for item in component.evidence)
    assert "Upstream Widgets Ltd" in component.copyright_text


def test_a_vendored_header_is_credited_to_the_artifact_that_includes_it(workspace):
    from isaaccapture_sbom import evidence as evidence_module

    space = evidence_module.discover(workspace.root, workspace.build)
    artifact = space.graph.by_name("_ext.cpython-312-x86_64-linux-gnu.so")[0]

    contributions = space.graph.contributions(artifact, space.ownership)

    assert "vendored:upstream-widgets-ltd" in contributions
    assert contributions["vendored:upstream-widgets-ltd"] == {"header-include"}


def test_the_codemodel_reader_works_on_a_codemodel_cmake_wrote(
    live_build_dir, checkout
):
    """Check real codemodel structure without depending on the current dependency set."""
    graph = discovery.BuildGraph(live_build_dir)

    assert graph.artifacts, "a configured build declares at least one artifact"
    assert any(
        graph.first_party(artifact, checkout) for artifact in graph.artifacts.values()
    ), "this project compiles some of its own sources"

    for output, artifact in graph.artifacts.items():
        assert output.is_absolute()
        assert graph.by_name(output.name), "every artifact is findable by name"
        for how in graph.contributions(artifact).values():
            assert how <= {"compiled-source", "header-include"}

    # Relative link paths must resolve to this build's indexed outputs.
    named = {path for node in graph._nodes.values() for path in node.linked}
    assert named & set(graph.artifacts), "this project links static libraries it builds"


def test_a_second_build_tree_is_not_mistaken_for_source(workspace, license_data):
    """A second configured build contains staged source copies that must not enter the source index."""
    from isaaccapture_sbom import evidence, licensing

    licensing.load_corpus(license_data)
    stray = workspace.root / "build-debug"
    (stray / "isaaccapture").mkdir(parents=True)
    (stray / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Debug\n")
    copy = stray / "isaaccapture" / "__init__.py"
    copy.write_text(
        (workspace.root / "src/python/isaaccapture/__init__.py").read_text()
    )

    discovered = evidence.discover(workspace.root, workspace.build)
    indexed = set(discovered.repo_files.by_hash.values())

    assert "./src/python/isaaccapture/__init__.py" in indexed
    assert not any(item.startswith("./build-debug/") for item in indexed)


def test_a_fetched_archive_records_where_it_came_from(workspace, license_data):
    """The fetch script preserves archive origin in a .source sidecar."""
    from isaaccapture_sbom import evidence, licensing

    licensing.load_corpus(license_data)
    archive = workspace.root / synth.SDK_ARCHIVE

    unknown = evidence.discover(workspace.root, workspace.build)
    before = unknown.components[synth.SDK_ARCHIVE]
    assert before.download_location == "NOASSERTION"
    assert before.supplier == "NOASSERTION"

    archive.with_name(archive.name + ".source").write_text(
        "https://api.ngc.nvidia.com/v2/resources/example/sdk.tar.gz\n"
    )
    known = evidence.discover(workspace.root, workspace.build)
    after = known.components[synth.SDK_ARCHIVE]

    assert after.download_location.startswith("https://api.ngc.nvidia.com/")
    assert after.supplier == "Organization: nvidia"
    assert "fetched from https://" in after.source_info


def test_a_build_tree_with_no_configuration_says_so(workspace):
    from isaaccapture_sbom import evidence

    cache = workspace.build / "CMakeCache.txt"
    cache.write_text(
        "\n".join(
            line
            for line in cache.read_text().splitlines()
            if not line.startswith("CMAKE_BUILD_TYPE:")
        )
        + "\n"
    )

    with pytest.raises(evidence.EvidenceError, match="CMAKE_BUILD_TYPE"):
        evidence.discover(workspace.root, workspace.build)


def test_the_supplier_is_read_from_the_project_not_written_here(workspace):

    assert discovery.project_supplier(workspace.root) == "Organization: NVIDIA"

    pyproject = workspace.root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            '{ name = "NVIDIA" }', '{ name = "Someone Else" }'
        ),
        encoding="utf-8",
    )

    assert discovery.project_supplier(workspace.root) == "Organization: Someone Else"


def test_a_project_naming_several_authors_stops_rather_than_choosing(workspace):
    """SPDX records one supplier; which one is not this tool's call to make."""

    pyproject = workspace.root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            '{ name = "NVIDIA" }',
            '{ name = "NVIDIA" }, { name = "Other Party" }',
        ),
        encoding="utf-8",
    )

    with pytest.raises(discovery.SupplierError, match="names 2 project authors"):
        discovery.project_supplier(workspace.root)


def test_a_relative_build_dir_still_keeps_the_staged_tree_out(
    workspace, license_data, monkeypatch
):
    """The workflow passes a relative build directory; staging must still be excluded."""
    from isaaccapture_sbom import evidence, licensing

    licensing.load_corpus(license_data)
    monkeypatch.chdir(workspace.root)

    discovered = evidence.discover(workspace.root, Path("build"))
    indexed = set(discovered.build_files.by_hash.values())

    assert indexed, "the build tree is indexed"
    assert not any("/python_package/" in item for item in indexed)


def test_a_dependency_installed_into_the_build_tree_is_not_vendored_source(
    workspace, license_data
):
    """Installed dependency headers must not become one vendored component per copyright holder."""
    from isaaccapture_sbom import evidence, licensing

    licensing.load_corpus(license_data)
    port = workspace.build / "vcpkg_installed/x64-linux/include/eigen/Core.h"
    assert port.is_file(), "the fixture no longer installs a port into the build tree"

    discovered = evidence.discover(workspace.root, workspace.build)
    holders = [key for key in discovered.components if key.startswith("vendored:")]

    assert not any("guennebaud" in key for key in holders), holders


def test_vendored_material_asserts_no_package_identifier(workspace, license_data):
    """Copyright ownership does not establish a resolvable package identity."""
    from isaaccapture_sbom import licensing

    licensing.load_corpus(license_data)
    source = workspace.root / "src" / "vendor" / "widget.hpp"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "// SPDX-FileCopyrightText: Copyright 2023 Upstream Widgets Ltd\n"
        "// SPDX-License-Identifier: BSL-1.0\n"
    )
    components, _ = discovery.discover_vendored(
        workspace.root, {source}, {"Example Org"}, tracked=()
    )
    component = next(iter(components.values()))

    assert component.kind == "vendored-source"
    assert component.purl == "NOASSERTION"
    assert component.supplier == "Organization: Upstream Widgets Ltd"


def _discover(root: Path, *candidates: str):
    """Discover over build inputs alone, as `discover_vendored` sees them."""
    components, _ = discovery.discover_vendored(
        root, {root / item for item in candidates}, {"NVIDIA"}, tracked=()
    )
    return components


def _vendored(root: Path, files: dict[str, str | bytes]):
    """Discover over files as a checkout tracking all of them would."""
    for relative, data in files.items():
        synth.write(root / relative, data)
    components, _ = discovery.discover_vendored(
        root, set(), {"NVIDIA"}, tracked=sorted(files)
    )
    return components


def _link(root: Path, link: str, target: str) -> None:
    (root / link).parent.mkdir(parents=True, exist_ok=True)
    (root / link).symlink_to(target)


def _notices(components) -> dict[str, set[str]]:
    return {key: set(c.copyright_text.splitlines()) for key, c in components.items()}


_LICENCE_TAG = "SPDX-" + "License-Identifier: MIT"
_IGNORED_BAR = "// REUSE-IgnoreStart\n// Copyright 2020 Bar Ltd\n// REUSE-IgnoreEnd\n"
_HEADERS = {
    "copyright-and-licence": f"// Copyright 2020 Bar Ltd\n// {_LICENCE_TAG}\n",
    "copyright-only": "// Copyright 2020 Bar Ltd\n",
    "unparsed-and-licence": f"// Copyright: 2020 Bar Ltd\n// {_LICENCE_TAG}\n",
    "unparsed-only": "// Copyright: 2020 Bar Ltd\n",
    "ignored-and-licence": f"{_IGNORED_BAR}// {_LICENCE_TAG}\n",
    "ignored-only": _IGNORED_BAR,
    "ignored-and-contributor": f"{_IGNORED_BAR}// SPDX-FileContributor: Someone\n",
}
_BAR = {"vendored:bar-ltd": {"Copyright 2020 Bar Ltd"}}
_BAR_UNPARSED = {"vendored:bar-ltd": {"Copyright: 2020 Bar Ltd"}}
_FOO = "Copyright 2020 Foo Inc"


_DECLARED = {
    "header": {
        "src/x.h": "// Copyright 2020 Foo Inc\n// SPDX-"
        + "License-Identifier: MIT OR Apache-2.0\n"
    },
    "sidecar": {
        "src/x.h": "// Copyright 2020 Foo Inc\n",
        "src/x.h.license": "SPDX-FileCopyrightText: Copyright 2020 Foo Inc\n"
        "SPDX-" + "License-Identifier: MIT OR Apache-2.0\n",
    },
    "REUSE.toml": {
        "src/x.h": "// Copyright 2020 Foo Inc\n",
        "REUSE.toml": synth.reuse_toml(
            synth.annotation("src/x.h", "Foo Inc", licence="MIT OR Apache-2.0")
        ),
    },
}


@pytest.mark.parametrize(
    "files", [pytest.param(files, id=name) for name, files in _DECLARED.items()]
)
def test_a_vendored_files_terms_are_what_reuse_declares(tmp_path, files):
    """All declaration forms preserve OR and supply every referenced text."""
    for relative, text in files.items():
        synth.write(tmp_path / relative, text)

    component = _discover(tmp_path, "src/x.h")["vendored:foo-inc"]

    assert component.license_concluded == "MIT OR Apache-2.0"
    assert component.license_declared == "MIT OR Apache-2.0"
    assert sorted(item.path for item in component.evidence) == [
        "Apache-2.0.txt",
        "MIT.txt",
    ]
    assert all(item.origin == "spdx-reference" for item in component.evidence)
    assert component.copyright_text == "Copyright 2020 Foo Inc"


@pytest.mark.parametrize(
    ("tag", "reason"),
    [
        ("", "no current SPDX-listed licence"),
        (
            "// SPDX-" + "License-Identifier: LicenseRef-Foo\n",
            "no current SPDX-listed licence",
        ),
        # MIT has a text; the LicenseRef would be defined by it in the document.
        (
            "// SPDX-" + "License-Identifier: MIT AND LicenseRef-Foo\n",
            "no current SPDX-listed licence with a reference text for LicenseRef-Foo",
        ),
        ("// SPDX-" + "License-Identifier: not ( an expression\n", "does not parse"),
        # A valid tag, then one quoted in the body: REUSE reads both.
        (
            "// SPDX-" + "License-Identifier: MIT\n"
            'const char *tag = "// SPDX-' + 'License-Identifier: GPL-2.0-only\\n";\n',
            "does not parse",
        ),
        (
            "// SPDX-" + "License-Identifier: MIT AND Foo-1.0\n",
            "Foo-1.0 is not an SPDX identifier",
        ),
        (
            "// SPDX-" + "License-Identifier: MIT AND Bar-2.0 AND Foo-1.0\n",
            "Bar-2.0, Foo-1.0 are not SPDX identifiers",
        ),
        # `+` is SPDX's operator, so the id is known; there is just no text for it.
        (
            "// SPDX-" + "License-Identifier: Apache-2.0+\n",
            "no current SPDX-listed licence with a reference text",
        ),
        # Listed, but as a licence: the exception slot takes only exceptions.
        (
            "// SPDX-" + "License-Identifier: MIT WITH Apache-2.0\n",
            "Apache-2.0 is not an SPDX exception",
        ),
        (
            "// SPDX-" + "License-Identifier: MIT WITH LicenseRef-Foo\n",
            "LicenseRef-Foo is not an SPDX exception",
        ),
        (
            "// SPDX-" + "License-Identifier: MIT AND rsync-linking-exception\n",
            "rsync-linking-exception is an SPDX exception, not a licence",
        ),
    ],
    ids=[
        "no-info",
        "licenseref-only",
        "licenseref-beside-listed",
        "unparseable",
        "quoted-in-body",
        "unknown-id",
        "unknown-ids",
        "or-later",
        "licence-as-exception",
        "licenseref-as-exception",
        "exception-as-licence",
    ],
)
def test_a_file_lending_nothing_is_named_not_covered_by_its_sibling(
    tmp_path, tag, reason
):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.h").write_text(
        "// Copyright 2020 Foo Inc\n// SPDX-" + "License-Identifier: MIT\n"
    )
    (tmp_path / "src/b.h").write_text("// Copyright 2020 Foo Inc\n" + tag)

    components = _discover(tmp_path, "src/a.h", "src/b.h")

    [(path, why)] = components["vendored:foo-inc"].unlicensed
    assert path == "src/b.h"
    assert reason in why


@pytest.mark.parametrize(
    "expression",
    ["BSL-1.0 WITH rsync-linking-exception", "GPL-2.0+"],
    ids=["exception-newer-than-grammar", "deprecated-or-later"],
)
def test_a_listed_identifier_passes_the_per_file_gate(tmp_path, expression):
    """Listed in the pinned SPDX data, whatever the grammar knows: an exception
    newer than it, or an id it maps through SPDX's `+` or-later operator."""
    (tmp_path / "x.h").write_text(
        "// Copyright 2020 Foo Inc\n// SPDX-" + f"License-Identifier: {expression}\n"
    )

    assert _discover(tmp_path, "x.h")["vendored:foo-inc"].unlicensed == ()


def test_a_link_without_reuse_info_takes_its_targets(tmp_path):
    """A symlink's own REUSE metadata wins; without it, use the target's metadata."""
    synth.write(tmp_path / "src/real/x.h", "// Copyright 2020 Foo Inc\n")
    _link(tmp_path, "src/link/x.h", "../real/x.h")
    synth.write(
        tmp_path / "REUSE.toml",
        synth.reuse_toml(synth.annotation("src/real/x.h", "Foo Inc")),
    )

    component = _discover(tmp_path, "src/link/x.h")["vendored:foo-inc"]

    assert component.unlicensed == ()
    assert component.license_concluded == "MIT"

    synth.write(
        tmp_path / "src/link/x.h.license",
        "SPDX-FileCopyrightText: Copyright 2020 Foo Inc\n"
        "SPDX-" + "License-Identifier: BSD-3-Clause\n",
    )
    component = _discover(tmp_path, "src/link/x.h")["vendored:foo-inc"]

    assert component.license_concluded == "BSD-3-Clause"


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(
            {
                "src/x.h": "#pragma once\n",
                "src/x.h.license": f"SPDX-FileCopyrightText: {_FOO}\n{_LICENCE_TAG}\n",
            },
            id="sidecar",
        ),
        pytest.param(
            {
                "assets/icon.svg": "<svg/>\n",
                "REUSE.toml": synth.reuse_toml(synth.annotation("assets/*", "Foo Inc")),
            },
            id="reuse-toml-svg",
        ),
        pytest.param(
            {
                "assets/data.json": "{}\n",
                "REUSE.toml": synth.reuse_toml(synth.annotation("assets/*", "Foo Inc")),
            },
            id="reuse-toml-json",
        ),
    ],
)
def test_a_holder_declared_beside_the_file_is_read(tmp_path, files):
    """A file of any type whose copyright only REUSE metadata states is still
    someone else's; the sidecar and REUSE.toml are not candidates themselves."""
    component = _vendored(tmp_path, files)["vendored:foo-inc"]

    [covered] = [name for name in files if not name.endswith((".license", ".toml"))]
    assert covered in component.source_info
    assert component.license_concluded == "MIT"
    assert component.copyright_text == _FOO


@pytest.mark.parametrize(
    ("header", "holder", "precedence", "expected"),
    [
        pytest.param(
            "copyright-and-licence",
            "Foo Inc",
            "override",
            {"vendored:foo-inc": {_FOO}},
            id="override-outranks-the-header",
        ),
        pytest.param(
            "copyright-only", None, "override", _BAR, id="override-licence-only"
        ),
        pytest.param(
            "copyright-and-licence",
            "Foo Inc",
            "closest",
            _BAR,
            id="closest-takes-a-full-header-first",
        ),
        pytest.param(
            "unparsed-only",
            "NVIDIA CORPORATION",
            "closest",
            {},
            id="closest-stands-in-for-a-file-reuse-finds-nothing-in",
        ),
        pytest.param(
            "unparsed-and-licence",
            "NVIDIA CORPORATION",
            "closest",
            _BAR_UNPARSED,
            id="closest-beside-a-licence-tag",
        ),
        pytest.param(
            "ignored-and-contributor",
            "NVIDIA CORPORATION",
            "closest",
            _BAR,
            id="closest-beside-a-contributor-line",
        ),
        *(
            pytest.param(
                header,
                "NVIDIA CORPORATION",
                "aggregate",
                _BAR_UNPARSED if header.startswith("unparsed") else _BAR,
                id=f"aggregate-project-{header}",
            )
            for header in (
                "copyright-and-licence",
                "copyright-only",
                "unparsed-and-licence",
                "ignored-and-licence",
                "unparsed-only",
                "ignored-only",
            )
        ),
        pytest.param(
            "copyright-and-licence",
            "Foo Inc",
            "aggregate",
            {"vendored:foo-inc": {_FOO, "Copyright 2020 Bar Ltd"}},
            id="aggregate-both-give-notices",
        ),
    ],
)
def test_a_header_counts_where_reuse_applies_it_beside_an_annotation(
    tmp_path, header, holder, precedence, expected
):
    """`override` sets the file aside; `aggregate` applies it beside the
    annotation whatever REUSE parsed; `closest` applies it where REUSE took
    anything from it, and stands in for it where REUSE found nothing."""
    toml = synth.reuse_toml(synth.annotation("src/x.h", holder, precedence=precedence))
    components = _vendored(tmp_path, {"src/x.h": _HEADERS[header], "REUSE.toml": toml})

    assert _notices(components) == expected


def test_declared_notices_naming_the_project_keep_the_file_first_party(tmp_path):
    components = _vendored(
        tmp_path,
        {
            "src/x.h": _HEADERS["copyright-only"],
            "src/x.h.license": "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA "
            "CORPORATION & AFFILIATES. All rights reserved.\n"
            "SPDX-" + "License-Identifier: Apache-2.0\n",
        },
    )

    assert components == {}


def test_a_file_with_several_declared_holders_is_grouped_by_one(tmp_path):
    """Grouped under the first other holder by name, whatever its case, with
    every notice but the project's as its copyright."""
    components = _vendored(
        tmp_path,
        {
            "src/x.h": "#pragma once\n",
            "src/x.h.license": "SPDX-FileCopyrightText: 2020 Zed Corp\n"
            "SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION\n"
            "SPDX-FileCopyrightText: 2021 acme Inc\n" + _LICENCE_TAG + "\n",
        },
    )

    assert _notices(components) == {
        "vendored:acme-inc": {"2020 Zed Corp", "2021 acme Inc"}
    }


def _png_with_text(keyword: bytes, text: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return len(data).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")

    header = (1).to_bytes(4, "big") * 2 + bytes([8, 0, 0, 0, 0])
    pixels = zlib.compress(b"\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"tEXt", keyword + b"\x00" + text)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )


def test_a_binary_file_is_not_scraped_for_a_holder(tmp_path):
    """A PNG's `Copyright` text chunk is not a header."""
    png = _png_with_text(b"Copyright", b"\nCopyright 2020 Foo Inc\n")
    assert discovery._COPYRIGHT_TAG.search(png.decode("utf-8", "replace"))

    assert _vendored(tmp_path, {"assets/logo.png": png}) == {}


def test_license_files_and_the_pool_are_not_candidates(tmp_path):
    """REUSE does not cover `LICENSES/` or `LICENSE*`/`COPYING*` files, and the
    holder one names is its template's or its licensor's."""
    upstream = "Copyright (c) 2020 Foo Inc\n\nPermission is hereby granted\n"
    components = _vendored(
        tmp_path,
        {
            "LICENSE": upstream,
            "COPYING.txt": upstream,
            "deps/foo/LICENSE.md": upstream,
            "LICENSES/MIT.txt": synth.spdx_text("MIT"),
        },
    )

    assert components == {}


def test_a_build_input_reuse_does_not_cover_is_not_a_candidate(tmp_path):
    synth.write(tmp_path / "inc/x.h", "// Copyright 2020 Foo Inc\n")
    synth.write(tmp_path / "inc/x.h.license", f"SPDX-FileCopyrightText: {_FOO}\n")
    synth.write(tmp_path / "inc/LICENSE", "Copyright (c) 2020 Bar Ltd\n")

    components = _discover(tmp_path, "inc/x.h", "inc/x.h.license", "inc/LICENSE")

    assert list(components) == ["vendored:foo-inc"]
    assert "Files: inc/x.h." in components["vendored:foo-inc"].source_info


@pytest.mark.parametrize(
    "relative",
    ["subprojects/foo/x.h", "gen/x.h", "sub/x.h"],
    ids=["meson-subproject", "vcs-ignored", "submodule"],
)
def test_a_build_input_the_project_does_not_cover_is_still_read(tmp_path, relative):
    """A compiled file ships whether or not REUSE lints it; a tracked one there is
    held to REUSE's coverage."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    synth.write(tmp_path / ".gitignore", "gen/\n")
    synth.write(tmp_path / ".gitmodules", '[submodule "sub"]\n\tpath = sub\n')
    synth.write(tmp_path / relative, "// Copyright 2020 Foo Inc\n")

    assert list(_discover(tmp_path, relative)) == ["vendored:foo-inc"]
    components, _ = discovery.discover_vendored(
        tmp_path, set(), {"NVIDIA"}, tracked=[relative]
    )
    assert components == {}


def test_the_files_the_build_read_are_named_first(tmp_path):
    for name in ("src/a.h", "src/z.h"):
        synth.write(tmp_path / name, "// Copyright 2020 Foo Inc\n")

    components, _ = discovery.discover_vendored(
        tmp_path, {tmp_path / "src/z.h"}, {"NVIDIA"}, tracked=["src/a.h", "src/z.h"]
    )

    assert "Files: src/z.h, src/a.h." in components["vendored:foo-inc"].source_info


def test_a_tracked_non_code_file_reaches_discovery(workspace, license_data):
    """Through the build's own entry point, not only `discover_vendored`."""
    from isaaccapture_sbom import evidence, licensing

    licensing.load_corpus(license_data)
    synth.write(workspace.root / "assets/vendor/icon.svg", "<svg/>\n")
    synth.write(
        workspace.root / "REUSE.toml",
        synth.reuse_toml(synth.annotation("assets/vendor/**", "Foo Inc")),
    )
    synth.track(workspace.root)

    discovered = evidence.discover(workspace.root, workspace.build)

    assert (
        "assets/vendor/icon.svg"
        in discovered.components["vendored:foo-inc"].source_info
    )


def test_a_copyright_string_in_a_first_party_body_names_no_holder(tmp_path):
    """REUSE reports every copyright line in a file; only the head's first one
    is the file's own."""
    components = _vendored(
        tmp_path,
        {
            "src/x.py": "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA "
            "CORPORATION & AFFILIATES. All rights reserved.\n"
            "# SPDX-" + "License-Identifier: Apache-2.0\n\n"
            'NOTICE = """\nCopyright 2019 Someone Else\n"""\n',
        },
    )

    assert components == {}


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            {
                "src/x.h": _HEADERS["copyright-only"],
                "src/x.h.license": _LICENCE_TAG + "\n",
            },
            _BAR,
            id="sidecar-licence-only",
        ),
        pytest.param(
            {"src/x.h": _HEADERS["unparsed-and-licence"]},
            _BAR_UNPARSED,
            id="line-reuse-does-not-parse",
        ),
        pytest.param(
            {"src/x.h": _HEADERS["ignored-and-licence"]},
            _BAR,
            id="inside-an-ignore-block",
        ),
        pytest.param(
            {
                "src/x.h": _HEADERS["copyright-only"],
                "src/x.h.license": f"SPDX-FileCopyrightText: 2020\n{_LICENCE_TAG}\n",
            },
            _BAR,
            id="declared-notice-naming-nobody",
        ),
    ],
)
def test_without_a_declared_holder_the_head_still_decides(tmp_path, files, expected):
    assert _notices(_vendored(tmp_path, files)) == expected


def test_the_head_is_read_in_the_encoding_it_is_written_in(tmp_path):
    header = _HEADERS["copyright-and-licence"].encode("utf-16")

    assert list(_vendored(tmp_path, {"src/x.h": header})) == ["vendored:bar-ltd"]


@pytest.mark.parametrize("tag", ["SPDX-FileCopyrightText", "SPDX-SnippetCopyrightText"])
def test_a_declared_notice_is_published_as_written(tmp_path, tag):
    components = _vendored(
        tmp_path,
        {
            "src/x.h": "#pragma once\n",
            "src/x.h.license": f"{tag}: 2020,2021 Foo Inc\n{_LICENCE_TAG}\n",
        },
    )

    assert components["vendored:foo-inc"].copyright_text == "2020,2021 Foo Inc"


@pytest.mark.parametrize(
    "sidecar",
    [f"SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION\n{_LICENCE_TAG}\n", ""],
    ids=["declaring", "empty"],
)
def test_an_aggregate_annotation_beside_a_sidecar_sets_the_file_aside(
    tmp_path, sidecar
):
    """REUSE reads the sidecar in the file's place, aggregate or not, and
    whether or not the sidecar says anything."""
    toml = synth.annotation("src/x.h", "NVIDIA CORPORATION", precedence="aggregate")
    components = _vendored(
        tmp_path,
        {
            "src/x.h": _HEADERS["copyright-and-licence"],
            "src/x.h.license": sidecar,
            "REUSE.toml": synth.reuse_toml(toml),
        },
    )

    assert components == {}


def test_a_nested_override_sets_the_file_aside_beneath_an_aggregate(tmp_path):
    """REUSE stops reading the file at any override, whatever lies above it."""
    nvidia = "NVIDIA CORPORATION"
    components = _vendored(
        tmp_path,
        {
            "src/x.h": _HEADERS["copyright-and-licence"],
            "REUSE.toml": synth.reuse_toml(
                synth.annotation("**", nvidia, precedence="aggregate")
            ),
            "src/REUSE.toml": synth.reuse_toml(
                synth.annotation("**", nvidia, precedence="override")
            ),
        },
    )

    assert components == {}


def test_a_link_takes_its_targets_holder_where_it_has_none(tmp_path):
    synth.write(tmp_path / "vendor/x.h", f"// {_LICENCE_TAG}\n")
    synth.write(
        tmp_path / "vendor/x.h.license", "SPDX-FileCopyrightText: 2020 Foo Inc\n"
    )
    _link(tmp_path, "src/x.h", "../vendor/x.h")

    component = _discover(tmp_path, "src/x.h")["vendored:foo-inc"]

    # REUSE gives the target no licence: its sidecar sets the header aside.
    assert component.license_concluded == "NOASSERTION"
    assert component.unlicensed


def test_a_link_takes_its_own_licence_and_its_targets_holder(tmp_path):
    synth.write(tmp_path / "vendor/x.h", _HEADERS["copyright-and-licence"])
    _link(tmp_path, "src/x.h", "../vendor/x.h")
    synth.write(
        tmp_path / "src/x.h.license", "SPDX-" + "License-Identifier: BSD-3-Clause\n"
    )

    component = _discover(tmp_path, "src/x.h")["vendored:bar-ltd"]

    assert component.license_concluded == "BSD-3-Clause"


@pytest.mark.parametrize(
    ("sidecar", "expected"),
    [
        pytest.param(
            "SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION\n"
            "SPDX-" + "License-Identifier: Apache-2.0\n",
            {},
            id="target-sidecar-names-the-project",
        ),
        pytest.param(
            f"SPDX-FileCopyrightText: {_FOO}\nSPDX-"
            + "License-Identifier: BSD-3-Clause\n",
            {"vendored:foo-inc": "BSD-3-Clause"},
            id="target-sidecar-names-another",
        ),
    ],
)
def test_a_link_reads_its_target_as_reuse_does(tmp_path, sidecar, expected):
    """The bytes a link shares with its target are read under the target's
    precedence: a header the target's sidecar sets aside decides nothing."""
    synth.write(tmp_path / "vendor/x.h", _HEADERS["copyright-and-licence"])
    synth.write(tmp_path / "vendor/x.h.license", sidecar)
    _link(tmp_path, "src/x.h", "../vendor/x.h")

    components = _discover(tmp_path, "src/x.h")

    assert {key: c.license_concluded for key, c in components.items()} == expected


@pytest.mark.parametrize("through", ["file", "link"])
@pytest.mark.parametrize(
    ("header", "holder", "precedence", "expected"),
    [
        pytest.param(
            "copyright-and-licence",
            "Foo Inc",
            "aggregate",
            {"vendored:foo-inc": "Apache-2.0 AND MIT"},
            id="aggregate",
        ),
        pytest.param(
            "unparsed-only",
            "NVIDIA CORPORATION",
            "aggregate",
            {"vendored:bar-ltd": "Apache-2.0"},
            id="aggregate-unparsed",
        ),
        pytest.param(
            "unparsed-and-licence",
            "NVIDIA CORPORATION",
            "closest",
            {"vendored:bar-ltd": "MIT"},
            id="closest",
        ),
    ],
)
def test_a_link_with_its_own_annotation_reads_as_a_file_in_its_place(
    tmp_path, through, header, holder, precedence, expected
):
    annotation = synth.annotation(
        "src/**", holder, licence="Apache-2.0", precedence=precedence
    )
    synth.write(tmp_path / "REUSE.toml", synth.reuse_toml(annotation))
    if through == "link":
        synth.write(tmp_path / "vendor/x.h", _HEADERS[header])
        _link(tmp_path, "src/x.h", "../vendor/x.h")
    else:
        synth.write(tmp_path / "src/x.h", _HEADERS[header])

    components = _discover(tmp_path, "src/x.h")

    assert {key: c.license_concluded for key, c in components.items()} == expected


def test_a_tracked_link_is_not_a_candidate(tmp_path):
    """REUSE skips links; only a build input is judged by its target."""
    synth.write(tmp_path / "vendor/x.h", _HEADERS["copyright-and-licence"])
    _link(tmp_path, "src/x.h", "../vendor/x.h")

    components, _ = discovery.discover_vendored(
        tmp_path, set(), {"NVIDIA"}, tracked=["src/x.h", "vendor/x.h"]
    )

    assert "Files: vendor/x.h." in components["vendored:bar-ltd"].source_info


def test_a_link_keeps_its_own_holder_where_its_target_has_no_info(tmp_path):
    synth.write(tmp_path / "vendor/x.h", "#pragma once\n")
    _link(tmp_path, "src/x.h", "../vendor/x.h")
    synth.write(tmp_path / "src/x.h.license", "SPDX-FileCopyrightText: 2020 Foo Inc\n")

    assert list(_discover(tmp_path, "src/x.h")) == ["vendored:foo-inc"]


def test_a_link_read_as_its_target_takes_the_targets_precedence(tmp_path):
    synth.write(tmp_path / "vendor/x.h", _HEADERS["unparsed-only"])
    synth.write(
        tmp_path / "REUSE.toml",
        synth.reuse_toml(
            synth.annotation("vendor/x.h", "NVIDIA CORPORATION", precedence="aggregate")
        ),
    )
    _link(tmp_path, "src/x.h", "../vendor/x.h")

    assert list(_discover(tmp_path, "src/x.h")) == ["vendored:bar-ltd"]


@pytest.mark.parametrize(
    ("name", "head"),
    [
        ("icon.svg", "<!-- Copyright 2020 Bar Ltd -->\n<svg/>\n"),
        ("bang.svg", "<!-- Copyright 2020 Bar Ltd --!>\n<svg/>\n"),
        ("x.c", "/* Copyright 2020 Bar Ltd */\nint x;\n"),
    ],
    ids=["html", "html-bang-end", "c"],
)
def test_comment_markup_is_not_part_of_a_scraped_holder(tmp_path, name, head):
    component = _vendored(tmp_path, {f"src/{name}": head})["vendored:bar-ltd"]

    assert component.name == "Vendored source (Bar Ltd)"
    assert component.copyright_text == "Copyright 2020 Bar Ltd"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the build left behind is the only dependency declaration there is."""

from __future__ import annotations

import pytest
import synth
from sbom import discovery


def test_source_trees_are_found_without_being_listed(workspace):
    components = discovery.discover_source_trees(workspace.build / "_deps")

    # license-list-data is the SPDX reference data deps/third_party fetches. It is
    # fetched the same way as everything else, so it is discovered the same way.
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
    assert alpha.purl.startswith("pkg:github/example-org/alpha")
    assert alpha.version == "v1.2.3", "the declared constraint stands in for a commit"


def test_no_checkout_on_the_clone_line_is_not_a_checkout(workspace):
    """`git clone --no-checkout <url>` must not be read as `git checkout <url>`."""
    beta = discovery.discover_source_trees(workspace.build / "_deps")["beta"]

    assert not beta.version.startswith("http")


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


def test_a_build_tree_without_a_file_api_reply_says_so(workspace):
    """The query lives in CMakeLists, so a missing reply means the tree was
    configured by a CMake too old to honour it — say that, don't guess."""
    import shutil

    shutil.rmtree(workspace.build / ".cmake")

    with pytest.raises(discovery.FileApiError, match="cmake_file_api"):
        discovery.BuildGraph(workspace.build, synth.CONFIG)


def test_the_graph_is_read_from_the_codemodel_not_from_generator_output(workspace):
    """link.txt is a Makefile-generator artifact; Ninja writes none. Reading the
    file API is what makes the collector work under either."""
    assert not list(workspace.build.rglob("link.txt"))
    assert not (workspace.build / "compile_commands.json").exists()

    graph = discovery.BuildGraph(workspace.build, synth.CONFIG)

    assert graph.by_name("_ext.cpython-312-x86_64-linux-gnu.so")
    assert graph.by_name("libgamma_renamed.so")


def test_third_party_checked_in_here_is_still_third_party(workspace):
    """Not everything third-party arrives through FetchContent. A file checked
    into the repository that someone else wrote says so in its copyright line,
    and that is what separates it from the project's own code -- not the
    directory it sits in."""
    from sbom import evidence as evidence_module

    space = evidence_module.discover(workspace.root, workspace.build)
    vendored = {
        key: component
        for key, component in space.components.items()
        if component.kind == "vendored-source"
    }

    assert [c.name for c in vendored.values()] == ["Upstream Widgets Ltd"]
    component = next(iter(vendored.values()))
    assert component.license_concluded == "BSL-1.0"
    # The file states an identifier and carries no text of its own, so the text
    # comes from the SPDX list this build pinned -- never from this repository's
    # own REUSE pool, whose copies license this project and carry its notices.
    assert [item.path for item in component.evidence] == ["BSL-1.0.txt"]
    assert all(item.origin == "spdx-reference" for item in component.evidence)
    # The notice the component states about itself is recorded, not dropped.
    assert "Upstream Widgets Ltd" in component.copyright_text


def test_a_vendored_header_is_credited_to_the_artifact_that_includes_it(workspace):
    from sbom import evidence as evidence_module

    space = evidence_module.discover(workspace.root, workspace.build)
    artifact = space.graph.by_name("_ext.cpython-312-x86_64-linux-gnu.so")[0]

    contributions = space.graph.contributions(artifact, space.ownership)

    assert "vendored:upstream-widgets-ltd" in contributions
    assert contributions["vendored:upstream-widgets-ltd"] == {"header-include"}


def test_the_codemodel_reader_works_on_a_codemodel_cmake_wrote(
    live_build_dir, checkout
):
    """Reads this repository's own build tree, not a synthesized one.

    Deliberately makes no claim about which dependencies are present -- only
    about the shape of what the reader gets back, so a bump cannot break it.
    """
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


def test_a_second_build_tree_is_not_mistaken_for_source(workspace, license_data):
    """A developer's other build directory holds staged copies of the repo.

    Walking those as source makes a file's origin ambiguous, which surfaces as
    an unexplained wheel member rather than a wrong one. A configured tree says
    what it is with a CMakeCache.txt.
    """
    from sbom import evidence, licensing

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
    """Archives here are downloaded at configure time, not checked in.

    The fetch script writes the URL beside the tarball, so the origin is read
    rather than guessed; without that file the document must say it does not
    know.
    """
    from sbom import evidence, licensing

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
    assert after.supplier == "Organization: api.ngc.nvidia.com"
    assert "fetched from https://" in after.source_info


def test_a_build_tree_with_no_configuration_says_so(workspace):
    """Guessing Release would read a directory that need not exist, and every
    member it would have explained becomes an unexplained one instead."""
    from sbom import evidence

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
    """A fork or a rename must not leave the document naming the wrong party."""
    from sbom import discovery

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
    from sbom import discovery

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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conditions that must stop a wheel from being published."""

from __future__ import annotations

import json
import shutil
import zipfile

import pytest
import synth
from sbom import build as build_module
from sbom import licensing
from sbom import wheelfile
from sbom import validate as validate_module

DIST_INFO = synth.DIST_INFO
WHEEL_NAME = synth.WHEEL_NAME


def _rebuild(workspace, payload):
    synth.write_wheel(workspace.wheel, payload)
    return build_module.build(
        workspace.root, workspace.build, workspace.wheel, workspace.root / "sbom"
    )


def test_a_member_from_nowhere_blocks_the_build(workspace):
    """A file that no build step and no fetched input explains is a coverage hole."""
    payload = synth.wheel_payload(workspace)
    payload["isaaccapture/mystery/libmystery.so.7"] = synth.elf_shared_object(
        "libmystery.so.7"
    )

    with pytest.raises(build_module.BuildError, match=r"libmystery\.so\.7"):
        _rebuild(workspace, payload)


def test_a_component_with_no_license_text_blocks_the_build(workspace):
    (workspace.build / "_deps/alpha-src/LICENSE").unlink()

    with pytest.raises(build_module.BuildError, match="alpha"):
        _rebuild(workspace, synth.wheel_payload(workspace))


def test_an_empty_license_file_does_not_count_as_evidence(workspace):
    # delta reaches the wheel through a static archive, so emptying its license
    # leaves the wheel's contents untouched and isolates the evidence gate.
    (workspace.build / "_deps/delta-src/BSD-LICENSE").write_text(
        "   \n", encoding="utf-8"
    )

    with pytest.raises(build_module.BuildError, match="delta"):
        _rebuild(workspace, synth.wheel_payload(workspace))


def test_a_library_the_host_cannot_resolve_is_not_claimed_as_the_host_s(workspace):
    """Resolving the SONAME is the evidence for "the machine supplied this".
    Without it the member is unexplained, and must be reported that way rather
    than attributed to a library nobody found."""
    payload = synth.wheel_payload(workspace)
    payload["isaaccapture.libs/libunknown-a1b2c3.so"] = synth.elf_shared_object(
        "libunknown.so.1", ("libc.so.6",)
    )
    synth.write_wheel(workspace.wheel, payload)

    with pytest.raises(build_module.BuildError, match="could not be traced"):
        build_module.build(
            workspace.root,
            workspace.build,
            workspace.wheel,
            workspace.root / "sbom",
            system_resolver=lambda soname: {
                "soname": soname,
                "resolved_path": None,
                "package": None,
                "copyright": None,
            },
        )


def test_the_build_host_is_the_last_resort_not_a_directory_name(workspace):
    """A library the build produced must be attributed to the build even when it
    sits in a repair tool's directory: the host lookup runs only once every other
    index has failed to explain the bytes."""
    produced = (workspace.build / "lib/libgamma_renamed.so").read_bytes()
    payload = synth.wheel_payload(workspace)
    payload["isaaccapture.libs/libgamma_renamed-abc123.so"] = produced
    staged = workspace.staged / "isaaccapture.libs"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "libgamma_renamed-abc123.so").write_bytes(produced)
    _rebuild(workspace, payload)

    stem = WHEEL_NAME.removesuffix(".whl")
    evidence = json.loads(
        (workspace.root / "sbom" / f"{stem}.build-evidence.json").read_text()
    )
    entry = next(
        item
        for item in evidence["attributions"]
        if item["path"] == "isaaccapture.libs/libgamma_renamed-abc123.so"
    )

    assert entry["origin"] == "built"
    assert entry["primary"] == "gamma"


def test_a_vendored_library_is_attributed_to_the_package_that_supplied_it(
    workspace, tmp_path
):
    copyright_file = tmp_path / "copyright"
    copyright_file.write_text(synth.spdx_text("Zlib"), encoding="utf-8")
    payload = synth.wheel_payload(workspace)
    payload["isaaccapture.libs/libz-a1b2c3.so"] = synth.elf_shared_object(
        "libz.so.1", ("libc.so.6",)
    )
    synth.write_wheel(workspace.wheel, payload)

    build_module.build(
        workspace.root,
        workspace.build,
        workspace.wheel,
        workspace.root / "sbom",
        system_resolver=lambda soname: {
            "soname": soname,
            "resolved_path": "/usr/lib/libz.so.1.3",
            "package": "zlib1g",
            # dpkg's maintainer, not the package name: `zlib1g` names the thing,
            # not whoever supplied it.
            "supplier": "Organization: Ubuntu Developers (ubuntu-devel@example.com)",
            "copyright": str(copyright_file),
        },
    )

    report = json.loads(
        (
            workspace.root / "sbom" / f"{WHEEL_NAME.removesuffix('.whl')}.licenses.json"
        ).read_text()
    )
    vendored = next(
        item for item in report["components"] if item["component"] == "system:libz.so.1"
    )
    assert vendored["license_concluded"] == "Zlib"
    assert vendored["supplier"] == (
        "Organization: Ubuntu Developers (ubuntu-devel@example.com)"
    )


def _replace_member(wheel, name, data):
    with zipfile.ZipFile(wheel) as archive:
        payload = {item: archive.read(item) for item in archive.namelist()}
    payload[name] = data
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(payload):
            archive.writestr(item, payload[item])


def _drop_member(wheel, name):
    with zipfile.ZipFile(wheel) as archive:
        payload = {
            item: archive.read(item) for item in archive.namelist() if item != name
        }
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(payload):
            archive.writestr(item, payload[item])


def test_editing_a_file_after_the_fact_fails_verification(built):
    _replace_member(built["wheel"], "isaaccapture/__init__.py", b"# tampered\n")

    failures = validate_module.check(built["wheel"])

    assert any("SHA256 mismatch" in failure for failure in failures)
    assert any("RECORD digest mismatch" in failure for failure in failures)


def test_adding_a_file_after_the_fact_fails_verification(built):
    _replace_member(built["wheel"], "isaaccapture/surprise.py", b"# added later\n")

    failures = validate_module.check(built["wheel"])

    assert any("not covered by the SBOM" in failure for failure in failures)


def test_mutating_the_wheel_breaks_its_manifest_binding(built):
    _replace_member(built["wheel"], "isaaccapture/__init__.py", b"# tampered\n")

    failures = validate_module.check(
        built["wheel"],
        build_evidence=built["evidence_path"],
        manifest=built["manifest_path"],
    )

    assert any("modified after its digest was bound" in failure for failure in failures)


def test_an_empty_member_verifies(built):
    """RECORD states size 0 for py.typed and namespace __init__.py."""
    with zipfile.ZipFile(built["wheel"]) as archive:
        empty = [item.filename for item in archive.infolist() if item.file_size == 0]

    assert empty, "fixture no longer carries an empty member"
    assert validate_module.check(built["wheel"]) == []


def test_another_tools_document_beside_ours_is_kept(built):
    """auditwheel ships `sboms/auditwheel.cdx.json`; PEP 770 shares that directory."""
    with zipfile.ZipFile(built["wheel"]) as archive:
        names = set(archive.namelist())

    assert f"{DIST_INFO}/sboms/{synth.AUDITWHEEL_SBOM}" in names
    assert validate_module.check(built["wheel"]) == []


def test_our_own_document_going_missing_is_rejected(built):
    _drop_member(built["wheel"], f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json")

    failures = validate_module.check(built["wheel"])

    assert any("embedded SBOM" in failure for failure in failures)


def test_duplicate_wheel_filenames_cannot_be_merged(tmp_path, built):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    for path in (first, second):
        path.write_text(json.dumps(built["manifest"]), encoding="utf-8")

    with pytest.raises(build_module.BuildError, match="published twice"):
        build_module.merge_manifests([first, second])


def _release_layout(built, tmp_path):
    """The shape publish-sbom-evidence assembles from the matrix artifacts."""
    wheels = tmp_path / "release" / "wheels"
    evidence = tmp_path / "release" / "evidence"
    wheels.mkdir(parents=True)
    evidence.mkdir(parents=True)
    (wheels / built["wheel"].name).write_bytes(built["wheel"].read_bytes())
    for sidecar in built["out_dir"].glob("*"):
        if sidecar.name != "manifest.json":
            (evidence / sidecar.name).write_bytes(sidecar.read_bytes())
    merged = tmp_path / "release" / "isaaccapture-sbom-manifest.json"
    merged.write_text(
        json.dumps(built["manifest"], indent=2, sort_keys=True), encoding="utf-8"
    )
    return merged, wheels, evidence


def test_a_complete_release_evidence_set_verifies(built, tmp_path):
    merged, wheels, evidence = _release_layout(built, tmp_path)

    assert validate_module.check_set(merged, wheels, evidence) == []


def test_a_missing_sidecar_fails_the_release_set(built, tmp_path):
    merged, wheels, evidence = _release_layout(built, tmp_path)
    next(evidence.glob("*.licenses.md")).unlink()

    failures = validate_module.check_set(merged, wheels, evidence)

    assert any("is missing" in failure for failure in failures)


def test_a_missing_wheel_fails_the_release_set(built, tmp_path):
    merged, wheels, evidence = _release_layout(built, tmp_path)
    (wheels / built["wheel"].name).unlink()

    failures = validate_module.check_set(merged, wheels, evidence)

    assert any("is missing from" in failure for failure in failures)


def test_missing_spdx_license_data_stops_the_build(workspace):
    """The reference texts come from the build; without them, nothing publishes."""
    shutil.rmtree(workspace.build / "_deps/license-list-data-src")

    with pytest.raises(licensing.CorpusError, match="SPDX license list data not found"):
        _rebuild(workspace, synth.wheel_payload(workspace))


def test_a_malformed_reuse_tag_does_not_reach_the_document(workspace):
    """A hand-written tag can be unparseable; emitting it verbatim would fail
    the published SBOM's own conformance check with no local signal."""
    payload = synth.wheel_payload(workspace)
    good = "# SPDX-" + "License-Identifier: mit or Apache-2.0\n"
    bad = "# SPDX-" + "License-Identifier: !!! not an expression\n"
    for name, text in (("tagged_ok.py", good), ("tagged_bad.py", bad)):
        payload[f"isaaccapture/{name}"] = text.encode()
        # Authored sources, staged into the package like any other .py.
        (workspace.root / "src/python/isaaccapture" / name).write_text(text)
        (workspace.staged / "isaaccapture" / name).write_text(text)
    synth.track(workspace.root)
    _rebuild(workspace, payload)

    with zipfile.ZipFile(workspace.wheel) as archive:
        spdx = json.loads(archive.read(f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json"))
    concluded = {
        item["fileName"].removeprefix("./"): item["licenseConcluded"]
        for item in spdx["files"]
    }

    # Normalised through the SPDX grammar, not passed through as written.
    assert concluded["isaaccapture/tagged_ok.py"] == "MIT OR Apache-2.0"
    assert concluded["isaaccapture/tagged_bad.py"] == "NOASSERTION"


def test_building_twice_on_the_same_wheel_is_refused(built):
    """A second pass would duplicate members rather than replace them."""
    with pytest.raises(build_module.BuildError, match="already carries"):
        build_module.build(
            built["workspace"].root,
            built["workspace"].build,
            built["wheel"],
            built["out_dir"],
        )


def test_a_name_match_a_build_id_contradicts_is_refused(workspace, license_data):
    """Two SDKs can ship a library under the same file name.

    The patched copy no longer hashes to either, so only the name is left to go
    on -- and a differing build-id says it is the wrong one. Claiming it would
    hand this binary the other component's licence.
    """
    from sbom import licensing

    licensing.load_corpus(license_data)
    payload = synth.wheel_payload(workspace)
    # Same basename as the member inside the SDK archive, different build-id,
    # and patched so the content matches nothing.
    payload["isaaccapture/other_sdk/libdemo.so"] = synth.elf_shared_object(
        "libdemo.so",
        ("libc.so.6", "libm.so.6"),
        build_id="2222222222222222222222222222222222222222",
    )
    synth.write_wheel(workspace.wheel, payload)

    with pytest.raises(build_module.BuildError, match="could not be traced"):
        build_module.build(
            workspace.root,
            workspace.build,
            workspace.wheel,
            workspace.root / "sbom",
        )


def test_an_unreadable_archive_says_so(workspace, license_data):
    """Only archives this build fetched are indexed, so one that will not open
    is a broken download -- not a stray file to walk past."""
    from sbom import discovery, licensing

    licensing.load_corpus(license_data)
    (workspace.root / synth.SDK_ARCHIVE).write_bytes(b"not an archive at all")

    with pytest.raises(discovery.ArchiveError, match="cannot be read"):
        build_module.build(
            workspace.root,
            workspace.build,
            workspace.wheel,
            workspace.root / "sbom",
        )


def test_bytes_more_than_one_component_holds_name_no_single_source(
    workspace, license_data
):
    """A stock licence text is the same file in every project that ships one.

    Naming whichever was indexed first would file one project's redistribution
    obligation under another's. Say the bytes are shared and claim no single
    source instead.
    """
    from sbom import licensing

    licensing.load_corpus(license_data)
    shared = synth.spdx_text("Apache-2.0").encode()
    for key in ("alpha", "beta"):
        (workspace.build / "_deps" / f"{key}-src").mkdir(parents=True, exist_ok=True)
        (workspace.build / "_deps" / f"{key}-src" / "LICENSE").write_bytes(shared)

    payload = synth.wheel_payload(workspace)
    payload["isaaccapture/viz/SOMEDEP_LICENSE"] = shared
    synth.write_wheel(workspace.wheel, payload)

    build_module.build(
        workspace.root, workspace.build, workspace.wheel, workspace.root / "sbom"
    )
    with zipfile.ZipFile(workspace.wheel) as archive:
        spdx = json.loads(archive.read(f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json"))
    entry = next(
        item for item in spdx["files"] if item["fileName"].endswith("SOMEDEP_LICENSE")
    )

    assert "shared-content" in entry["comment"]
    assert "more than one component" in entry["comment"]
    # Both candidates named, neither claimed as the source.
    assert "alpha" in entry["comment"] and "beta" in entry["comment"]


def test_the_excluded_list_cannot_exempt_a_smuggled_file(built):
    """The list lives in the document being checked.

    Treating it as files to skip lets whoever edits the document exempt anything
    they add to the wheel from the only coverage check a consumer runs.
    """
    with zipfile.ZipFile(built["wheel"]) as archive:
        payload = {item: archive.read(item) for item in archive.namelist()}
    # write_wheel regenerates RECORD; keeping the old one would make a duplicate
    # member, which is a different finding and now refused outright.
    payload.pop(f"{DIST_INFO}/RECORD", None)

    payload["isaaccapture/_backdoor.py"] = b"import os\n"
    document = f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json"
    spdx = json.loads(payload[document])
    for package in spdx["packages"]:
        if package["SPDXID"] == "SPDXRef-Package-wheel":
            package["packageVerificationCode"][
                "packageVerificationCodeExcludedFiles"
            ].append("./isaaccapture/_backdoor.py")
    payload[document] = json.dumps(spdx, indent=2, sort_keys=True).encode() + b"\n"
    synth.write_wheel(built["wheel"], payload)

    failures = validate_module.check(built["wheel"])

    assert any("excluded-files list must be exactly" in item for item in failures)
    assert any("not covered by the SBOM" in item for item in failures)


def test_two_members_under_one_name_are_refused(built):
    """Reading by name yields the last copy, so the first is hashed by nothing.

    Which one a consumer extracts is up to their unzip, so an archive carrying
    both is not describable and must not be accepted.
    """
    with zipfile.ZipFile(built["wheel"]) as archive:
        names = [item for item in archive.namelist() if not item.endswith("/")]
        payload = {item: archive.read(item) for item in names}

    smuggled = built["wheel"].with_name("duplicated.whl")
    with zipfile.ZipFile(smuggled, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            if name == "isaaccapture/__init__.py":
                archive.writestr(name, b"import os\n")
            archive.writestr(name, payload[name])

    with pytest.raises(wheelfile.WheelError, match="more than one member"):
        validate_module.check(smuggled)


def test_one_discovery_serves_every_wheel_without_bleeding_between_them(
    workspace, tmp_path
):
    """The build tree is read once for a run; a wheel's own findings are its own.

    `_from_build_host` adds the system libraries a wheel turned out to need. If
    those land on the shared `Discovery`, the next wheel's evidence inherits
    components it does not ship; if they land nowhere the evidence reaches, the
    document names licence texts the evidence cannot account for and the wheel
    fails its own verification.
    """
    from sbom import evidence as evidence_module

    copyright_file = tmp_path / "copyright"
    copyright_file.write_text(synth.spdx_text("Zlib"), encoding="utf-8")
    payload = synth.wheel_payload(workspace)
    payload["isaaccapture.libs/libz-a1b2c3.so"] = synth.elf_shared_object(
        "libz.so.1", ("libc.so.6",)
    )
    synth.write_wheel(workspace.wheel, payload)

    plain = workspace.wheel.parent / "plain-1.0-py3-none-any.whl"
    synth.write_wheel(plain, synth.wheel_payload(workspace))

    licensing.load_corpus(workspace.build / "_deps" / "license-list-data-src" / "json")
    discovery = evidence_module.discover(workspace.root, workspace.build)
    resolver = lambda soname: {  # noqa: E731
        "soname": soname,
        "resolved_path": "/usr/lib/libz.so.1.3",
        "package": "zlib1g",
        "supplier": "Organization: Ubuntu Developers (ubuntu-devel@example.com)",
        "copyright": str(copyright_file),
    }
    for wheel in (workspace.wheel, plain):
        build_module.build(
            workspace.root,
            workspace.build,
            wheel,
            workspace.root / "sbom",
            system_resolver=resolver,
            discovery=discovery,
        )

    def components(wheel_name):
        stem = wheel_name.removesuffix(".whl")
        sidecar = workspace.root / "sbom" / f"{stem}.build-evidence.json"
        return json.loads(sidecar.read_text())["components"]

    assert "system:libz.so.1" in components(WHEEL_NAME), (
        "the wheel's own evidence must account for what its members needed"
    )
    assert "system:libz.so.1" not in components(plain.name), (
        "a wheel that ships no such library must not inherit it"
    )
    assert validate_module.check(workspace.wheel) == []

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End to end: a repaired wheel goes in, a publishable wheel comes out."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile

import synth
from sbom import build as build_module
from sbom import validate as validate_module
from sbom import wheelfile

DIST_INFO = synth.DIST_INFO
WHEEL_NAME = synth.WHEEL_NAME


def _names(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def _embedded(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return json.loads(archive.read(f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json"))


def test_wheel_verifies_against_its_own_sbom(built):
    assert validate_module.check(built["wheel"]) == []


def test_release_validation_passes(built):
    failures = validate_module.check(
        built["wheel"],
        build_evidence=built["evidence_path"],
        manifest=built["manifest_path"],
    )
    assert failures == []


def test_sbom_lands_in_dist_info_sboms_beside_any_other_tools(built):
    sboms = {name for name in _names(built["wheel"]) if f"{DIST_INFO}/sboms/" in name}

    # PEP 770 shares the directory: auditwheel wrote its document before us.
    assert sboms == {
        f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json",
        f"{DIST_INFO}/sboms/{synth.AUDITWHEEL_SBOM}",
    }


def test_every_wheel_file_is_covered(built):
    spdx = _embedded(built["wheel"])
    documented = {item["fileName"].removeprefix("./") for item in spdx["files"]}
    excluded = {
        item.removeprefix("./")
        for package in spdx["packages"]
        for item in package.get("packageVerificationCode", {}).get(
            "packageVerificationCodeExcludedFiles", []
        )
    }

    assert _names(built["wheel"]) - documented == excluded


def test_every_member_records_where_it_came_from(built):
    evidence = json.loads(built["evidence_path"].read_text(encoding="utf-8"))
    origins = {item["path"]: item["origin"] for item in evidence["attributions"]}

    assert origins["isaaccapture/__init__.py"] == "repo-source"
    assert origins["isaaccapture/_generated.py"] == "generated"
    assert origins["isaaccapture/_ext.cpython-312-x86_64-linux-gnu.so"] == "built"
    assert origins["isaaccapture/viz/libgamma_renamed.so"] == "built"
    assert origins["isaaccapture/viz/GAMMA_LICENSE"] == "copied"
    assert origins["isaaccapture/sdk/native/libdemo.so"] == "archive-copy"
    assert origins["vendorpy/mod.py"] == "copied"


def test_post_processed_files_are_recorded_as_derived_not_as_copies(built):
    """patchelf and the MJCF rewrite change bytes; the claim has to change too."""
    evidence = json.loads(built["evidence_path"].read_text(encoding="utf-8"))
    origins = {item["path"]: item for item in evidence["attributions"]}

    patched = origins["isaaccapture/sdk/native/libextra.so"]
    assert patched["origin"] == "archive-derived"
    assert "modified copy of" in patched["detail"]

    rewritten = origins["vendorpy/assets/hand.xml"]
    assert rewritten["origin"] == "derived"


def test_components_are_discovered_not_declared(built):
    spdx = _embedded(built["wheel"])
    names = {package["name"] for package in spdx["packages"]}

    assert {"alpha", "beta", "gamma", "delta", "vendorpy"} <= names
    assert "demosdk-1.2.0-linux-amd64" in names
    assert "omega" not in names, "a test-only dependency is not in the wheel"


def test_a_dependency_reaching_no_wheel_file_is_recorded_as_excluded(built):
    report = json.loads(built["report_path"].read_text(encoding="utf-8"))
    excluded = [item["component"] for item in report["exclusions"]]

    assert "omega" in excluded, "a test-only dependency ships nothing"
    # The SPDX reference data is fetched like any other dependency, so it is
    # inventoried like any other dependency -- and excluded for the same reason.
    assert "license-list-data" in excluded


def test_licenses_are_packaged_and_declared(built):
    names = _names(built["wheel"])
    third_party = {
        name for name in names if f"{DIST_INFO}/licenses/third-party/" in name
    }

    assert f"{DIST_INFO}/licenses/THIRD-PARTY-NOTICES.md" in names
    assert any("/alpha/" in name for name in third_party)
    assert any("demosdk" in name for name in third_party)

    raw, metadata = wheelfile.read_metadata(built["wheel"], DIST_INFO)
    declared = set(metadata.get("license_files") or [])
    # PEP 639 states these relative to .dist-info/licenses/, so each must
    # resolve back to a member that is actually in the wheel.
    prefix = f"{DIST_INFO}/licenses/"
    assert {name.removeprefix(prefix) for name in third_party} <= declared
    for value in declared:
        assert f"{prefix}{value}" in names, (
            f"License-File {value!r} resolves to nothing"
        )
    assert "LICENSE.md" in declared, (
        "the project's own license must survive the rewrite"
    )

    # No RFC 822 folding: a wrapped path is valid, and packaging unfolds it, but
    # nothing else writes one and a continuation line defeats simpler readers.
    assert not [
        line for line in raw.decode().splitlines() if line.startswith((" ", "\t"))
    ]


def test_an_unidentified_grant_travels_verbatim_in_the_document(built):
    spdx = _embedded(built["wheel"])
    extracted = {item["licenseId"]: item for item in spdx["hasExtractedLicensingInfos"]}

    # The id is minted from the component's key -- its path, here -- not from a
    # basename two archives in different directories would share.
    ref = next(item for item in extracted if "demosdk" in item)
    assert "deps" in ref and "vendor" in ref, ref
    assert "DEMO CORP" in extracted[ref]["extractedText"]


def test_reuse_tags_on_shipped_files_are_reported_per_file(built):
    spdx = _embedded(built["wheel"])
    tagged = {
        item["fileName"].removeprefix("./"): item.get("licenseConcluded")
        for item in spdx["files"]
    }

    assert tagged["vendorpy/mod.py"] == "Apache-2.0"
    assert tagged["isaaccapture/__init__.py"] == "Apache-2.0"


def test_record_is_consistent_after_the_rewrite(built):
    record = wheelfile.read_record(built["wheel"], DIST_INFO)
    with zipfile.ZipFile(built["wheel"]) as archive:
        members = set(archive.namelist())
        for name, (digest, size) in record.items():
            if name == f"{DIST_INFO}/RECORD":
                continue
            data = archive.read(name)
            assert digest == wheelfile.record_hash(data), name
            assert size == str(len(data)), name
    assert set(record) == members


def test_manifest_binds_the_final_wheel_digest(built):
    entry = built["manifest"]["wheels"][0]

    assert entry["filename"] == WHEEL_NAME
    assert entry["sha256"] == build_module.sha256_file(built["wheel"])
    assert entry["sbom_in_wheel"] == f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json"


def test_published_copy_matches_the_embedded_one(built):
    with zipfile.ZipFile(built["wheel"]) as archive:
        embedded = archive.read(f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json")

    assert built["spdx_path"].read_bytes() == embedded


def test_external_runtime_dependencies_are_not_claimed_as_contents(built):
    spdx = _embedded(built["wheel"])
    external = {
        package["name"]
        for package in spdx["packages"]
        if package["SPDXID"].startswith("SPDXRef-Package-external-")
    }

    assert {"libc.so.6", "libstdc++.so.6"} <= external
    assert "libdemo.so" not in external, "a library inside the wheel is not external"


def test_consumer_requirements_come_from_wheel_metadata(built):
    spdx = _embedded(built["wheel"])
    comments = " ".join(
        package.get("comment", "")
        for package in spdx["packages"]
        if package["SPDXID"].startswith("SPDXRef-Package-pypi-")
    )

    assert "numpy>=1.23.0" in comments
    assert "pyyaml>=6.0.3" in comments


def test_build_is_reproducible_under_source_date_epoch(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")

    digests = []
    for run in ("a", "b"):
        space = synth.create(tmp_path / run)
        synth.write_wheel(space.wheel, synth.wheel_payload(space))
        build_module.build(space.root, space.build, space.wheel, space.root / "sbom")
        digests.append(build_module.sha256_file(space.wheel))

    assert digests[0] == digests[1]


def test_source_date_epoch_normalizes_timestamps_the_input_wheel_carried(
    workspace, monkeypatch
):
    """A zip records a local timestamp per member, so a wheel carries its
    builder's clock. Under SOURCE_DATE_EPOCH the rewrite has to normalize the
    members it copies through as well, or it is reproducible only for an input
    wheel that already was."""
    import zipfile as zf

    stamped = synth.wheel_payload(workspace)
    with zf.ZipFile(workspace.wheel, "w", zf.ZIP_DEFLATED) as archive:
        for name in sorted(stamped):
            archive.writestr(
                zf.ZipInfo(name, date_time=(2021, 6, 5, 4, 3, 2)), stamped[name]
            )
        archive.writestr(zf.ZipInfo("x", date_time=(2021, 6, 5, 4, 3, 2)), b"")
    synth.write_wheel(workspace.wheel, stamped)

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    build_module.build(
        workspace.root, workspace.build, workspace.wheel, workspace.root / "sbom"
    )

    with zf.ZipFile(workspace.wheel) as archive:
        stamps = {info.date_time for info in archive.infolist()}
    assert len(stamps) == 1, f"members carry differing timestamps: {stamps}"


def test_components_sharing_a_licence_text_share_one_entry(workspace, license_data):
    """A LicenseRef identifies a text, so one text is one entry.

    Two components can carry byte-identical terms -- two builds of one SDK, a
    vendor licence applied across products -- and minting an id each duplicates
    the whole text in the document.
    """
    from sbom import licensing

    licensing.load_corpus(license_data)
    for key in ("alpha", "beta"):
        root = workspace.build / "_deps" / f"{key}-src"
        root.mkdir(parents=True, exist_ok=True)
        (root / "LICENSE").write_text(synth.PROPRIETARY_LICENSE)
    synth.track(workspace.root)

    payload = synth.wheel_payload(workspace)
    for key in ("alpha", "beta"):
        payload[f"isaaccapture/{key}_notice.txt"] = synth.PROPRIETARY_LICENSE.encode()
    synth.write_wheel(workspace.wheel, payload)
    build_module.build(
        workspace.root, workspace.build, workspace.wheel, workspace.root / "sbom"
    )

    with zipfile.ZipFile(workspace.wheel) as archive:
        spdx = json.loads(archive.read(f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json"))
    bodies = [item["extractedText"] for item in spdx["hasExtractedLicensingInfos"]]

    assert len(bodies) == len(set(bodies)), "a text is carried more than once"
    used = {
        ref
        for package in spdx["packages"]
        for field in ("licenseConcluded", "licenseDeclared")
        for ref in re.findall(r"LicenseRef-[\w.-]+", package.get(field, ""))
    }
    defined = {item["licenseId"] for item in spdx["hasExtractedLicensingInfos"]}
    assert used <= defined, f"undefined after sharing: {sorted(used - defined)}"


def test_a_requirement_this_run_also_built_points_at_its_own_document(
    workspace, license_data, tmp_path
):
    """A transition wheel requires the wheel beside it; that is not an unknown.

    Left as an ordinary requirement it reads as something an installer resolves,
    with the version, supplier and licence unasserted -- for a distribution this
    same build produced and described in full.
    """
    from sbom import cli

    synth.write_wheel(workspace.wheel, synth.wheel_payload(workspace))
    companion = workspace.wheel.parent / "isaacteleop-0.4.0-py3-none-any.whl"
    synth.write_wheel(
        companion,
        {
            "isaacteleop-0.4.0.dist-info/METADATA": "\n".join(
                [
                    "Metadata-Version: 2.4",
                    "Name: isaacteleop",
                    "Version: 0.4.0",
                    "License-Expression: Apache-2.0",
                    "Requires-Dist: isaaccapture>=0.4.0",
                    "",
                ]
            ).encode(),
            "isaacteleop-0.4.0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: setuptools\nRoot-Is-Purelib: true\n"
            ),
        },
    )

    out = workspace.root / "sbom"
    # Through the CLI: it is what orders the wheels, and a document may only
    # cite one that already exists.
    assert (
        cli.main(
            [
                "--repo-root",
                str(workspace.root),
                "build",
                "--build-dir",
                str(workspace.build),
                "--license-data",
                str(license_data),
                "--out-dir",
                str(out),
                "--wheel",
                str(companion),
                "--wheel",
                str(workspace.wheel),
            ]
        )
        == 0
    )

    spdx = json.loads((out / "isaacteleop-0.4.0-py3-none-any.spdx.json").read_text())
    assert not [
        package
        for package in spdx["packages"]
        if package["SPDXID"].startswith("SPDXRef-Package-pypi-isaaccapture")
    ], "the sibling is not restated as an install-time unknown"

    ref = spdx["externalDocumentRefs"][0]
    sibling = (out / f"{WHEEL_NAME.removesuffix('.whl')}.spdx.json").read_bytes()
    assert ref["spdxDocument"] == json.loads(sibling)["documentNamespace"]
    assert ref["checksum"] == {
        "algorithm": "SHA1",
        "checksumValue": hashlib.sha1(sibling).hexdigest(),  # noqa: S324
    }
    assert any(
        item["relatedSpdxElement"]
        == f"{ref['externalDocumentId']}:SPDXRef-Package-wheel"
        for item in spdx["relationships"]
    )
    assert validate_module.check(companion) == []


def _rewrite_document(wheel, mutate):
    """Put a modified SBOM back into a wheel, under its own name."""
    info = wheelfile.scan(wheel)
    name = wheelfile.sbom_member(info.dist_info, wheel.name)
    with zipfile.ZipFile(wheel) as archive:
        spdx = json.loads(archive.read(name))
    assert mutate(spdx), "nothing to mutate"
    body = json.dumps(spdx, indent=2, sort_keys=True).encode() + b"\n"
    staged = wheel.parent / f"staged-{wheel.name}"
    wheelfile.rewrite(
        wheel,
        staged,
        additions={},
        replacements={name: body},
        dist_info=info.dist_info,
    )
    staged.replace(wheel)


def test_a_components_terms_may_span_several_files(workspace, license_data):
    """A component can state its licence in more than one file.

    The LicenseRef then carries the join, which has no digest of its own, so the
    build states one. Holding the entry to a single file's digest instead failed
    a correct wheel.
    """
    from sbom import licensing

    licensing.load_corpus(license_data)
    root = workspace.build / "_deps" / "alpha-src"
    root.mkdir(parents=True, exist_ok=True)
    (root / "LICENSE").write_text(synth.PROPRIETARY_LICENSE)
    (root / "COPYING").write_text(synth.PROPRIETARY_LICENSE + "\nAnd a second file.")
    synth.track(workspace.root)
    synth.write_wheel(workspace.wheel, synth.wheel_payload(workspace))

    out = workspace.root / "sbom"
    build_module.build(workspace.root, workspace.build, workspace.wheel, out)

    stem = WHEEL_NAME.removesuffix(".whl")
    evidence_path = out / f"{stem}.build-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    texts = [
        item
        for item in evidence["components"]["alpha"]["evidence"]
        if item["kind"] in ("grant", "pool")
    ]
    assert len(texts) > 1, "the component was set up to state its terms twice"

    spdx = _embedded(workspace.wheel)
    joined = next(
        item
        for item in spdx["hasExtractedLicensingInfos"]
        if "alpha" in item["licenseId"]
    )
    assert joined["extractedText"].count(synth.PROPRIETARY_LICENSE) >= 1
    assert validate_module.check(workspace.wheel, build_evidence=evidence_path) == []


def test_a_rewritten_licence_ref_body_is_caught_however_it_was_minted(built):
    """The verbatim terms are what a LicenseRef stands for.

    A shared id belongs to no single component, which is not a reason to stop
    checking it: that is the id the proprietary SDK terms travel under.
    """
    swapped = "You may do whatever you like with this software."

    def rewrite(spdx):
        for item in spdx["hasExtractedLicensingInfos"]:
            item["extractedText"] = swapped
        return bool(spdx["hasExtractedLicensingInfos"])

    _rewrite_document(built["wheel"], rewrite)

    failures = validate_module.check(
        built["wheel"], build_evidence=built["evidence_path"]
    )
    assert any("not the terms the build read" in item for item in failures), failures

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End to end: a repaired wheel goes in, a publishable wheel comes out."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest
import synth
from isaaccapture_sbom import build as build_module
from isaaccapture_sbom import document as document_module
from isaaccapture_sbom import inventory as inventory_module
from isaaccapture_sbom import licensing
from isaaccapture_sbom import report as report_module
from isaaccapture_sbom import validate as validate_module
from isaaccapture_sbom import wheelfile

DIST_INFO = synth.DIST_INFO
WHEEL_NAME = synth.WHEEL_NAME


def _names(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def _embedded(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return json.loads(archive.read(f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json"))


def _wheel_package(spdx: dict) -> dict:
    return next(
        item for item in spdx["packages"] if item["SPDXID"] == "SPDXRef-Package-wheel"
    )


def _release_failures(built) -> list[str]:
    """What `check` says of the wheel with its build evidence and manifest."""
    return validate_module.check(
        built["wheel"],
        build_evidence=built["evidence_path"],
        manifest=built["manifest_path"],
    )


def test_wheel_verifies_against_its_own_sbom(built):
    assert validate_module.check(built["wheel"]) == []


def test_release_validation_passes(built):
    assert _release_failures(built) == []


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


def test_a_dependency_reaching_no_wheel_file_is_recorded_by_the_build(built):
    components = built["build-evidence"]["components"]

    assert components["omega"]["in_this_wheel"] is False, "a test-only dependency"
    # The SPDX reference data is fetched like any other dependency, so it is
    # inventoried like any other dependency -- and ships nothing for the same reason.
    assert components["license-list-data"]["in_this_wheel"] is False
    # What ships nothing is the build's to record, not the wheel's.
    assert "omega" not in json.dumps(built["licenses"])
    assert "omega" not in built["md"]


def test_where_the_license_list_was_read_is_left_to_the_build_evidence(built):
    evidence = built["build-evidence"]["license_data"]
    report = built["licenses"]["license_data"]

    assert "source" in evidence and "source" not in report
    assert report["commit"] == evidence["commit"]


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


@pytest.fixture
def eula_beside_mit(workspace):
    """alpha ships its MIT LICENSE and an EULA that matches nothing."""
    eula = synth.PROPRIETARY_LICENSE.replace("Demo SDK", "Alpha runtime")
    (workspace.build / "_deps" / "alpha-src" / "LICENSE-EULA.txt").write_text(
        eula, encoding="utf-8"
    )
    synth.track(workspace.root)
    return {**synth.collect_and_bind(workspace), "eula": eula}


def test_an_unmatched_grant_beside_a_matched_one_is_carried_and_reported(
    eula_beside_mit,
):
    built = eula_beside_mit
    package = next(
        item for item in built["spdx"]["packages"] if item["name"] == "alpha"
    )
    assert package["licenseConcluded"] == "LicenseRef-alpha AND MIT"
    extracted = {
        item["licenseId"]: item for item in built["spdx"]["hasExtractedLicensingInfos"]
    }
    assert extracted["LicenseRef-alpha"]["extractedText"] == built["eula"]
    assert extracted["LicenseRef-alpha"]["comment"] == (
        "Matched no text in the SPDX reference corpus; reproduced here verbatim "
        "from alpha-src/LICENSE-EULA.txt."
    )
    unidentified = {
        item["component"]: item
        for item in built["licenses"]["components_with_unidentified_license"]
    }
    assert unidentified["alpha"] == {
        "component": "alpha",
        "reason": "1 of its 2 shipped grants is carried verbatim rather than "
        "named: matched no text in the SPDX reference corpus",
        "evidence": ["alpha-src/LICENSE-EULA.txt"],
    }
    assert _release_failures(built) == []


def test_a_component_shipping_only_a_reuse_pool_is_a_reported_gap(workspace):
    """The pool passes the gate and concludes nothing, so it cannot fall
    between the lists and leave the summary saying every licence is known."""
    source = workspace.build / "_deps" / "alpha-src"
    (source / "LICENSES").mkdir()
    (source / "LICENSE").rename(source / "LICENSES" / "MIT.txt")
    synth.track(workspace.root)
    built = synth.collect_and_bind(workspace)

    report = built["licenses"]
    alpha = next(item for item in report["components"] if item["component"] == "alpha")
    assert alpha["license_concluded"] == "NOASSERTION"
    listing = {
        key: [item for item in report[key] if item["component"] == "alpha"]
        for key in report_module.GAP_LISTS
    }
    assert [key for key, found in listing.items() if found] == [
        "components_with_unresolved_license"
    ]
    [entry] = listing["components_with_unresolved_license"]
    assert entry["evidence"] == ["alpha-src/LICENSES/MIT.txt"]
    assert entry["reason"].startswith("only REUSE pool texts ship (MIT)")
    [record] = built["manifest"]["wheels"]
    assert record["components_with_unresolved_license"] == [entry]
    # The wheel's own conclusion cannot AND in terms nobody concluded.
    assert _wheel_package(built["spdx"])["licenseConcluded"] == "NOASSERTION"
    assert _release_failures(built) == []


def _report_of(**gaps) -> dict:
    """A report payload with one identified component and the given gap lists."""
    row = {
        "name": "alpha",
        "kind": "source-tree",
        "version": "1",
        "license_concluded": "MIT",
        "how_it_reaches_the_wheel": [],
        "evidence": [],
    }
    return {
        "wheel": WHEEL_NAME,
        "generated_from_commit": None,
        "spdx_license_list_version": None,
        "components": [row],
        "externally_supplied_runtime": [],
        "consumer_requirements": [],
        **{key: gaps.get(key, []) for key in report_module.GAP_LISTS},
    }


def test_an_identified_set_reports_no_gaps():
    rendered = report_module.report_markdown(_report_of())

    assert rendered.endswith(
        "## License evidence gaps\n\n"
        "None; every redistributed component ships an identified license.\n"
    )


@pytest.mark.parametrize("key", list(report_module.GAP_LISTS))
def test_any_gap_list_withholds_the_all_identified_summary(key):
    gap = {"component": "alpha", "reason": "why"}
    rendered = report_module.report_markdown(_report_of(**{key: [gap]}))

    assert f"{report_module.GAP_LISTS[key]}\n\n- `alpha` — why\n" in rendered
    assert "None; every redistributed component" not in rendered


def _rewrite_alpha(built, expression: str) -> list[str]:
    """Rewrite alpha's expressions in the embedded document; what check says."""
    member = f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json"
    payload = synth.members(built["wheel"])
    spdx = json.loads(payload[member])
    for package in spdx["packages"]:
        if package["name"] == "alpha":
            package["licenseConcluded"] = package["licenseDeclared"] = expression
    payload[member] = json.dumps(spdx).encode()
    synth.write_wheel(built["wheel"], payload)
    return validate_module.check(built["wheel"], build_evidence=built["evidence_path"])


def _disagreement(expression: str) -> list[str]:
    recorded = "LicenseRef-alpha AND MIT"
    return [
        f"alpha: the document says {field} {expression!r}, the build recorded "
        f"{recorded!r}"
        for field in ("licenseConcluded", "licenseDeclared")
    ]


@pytest.mark.parametrize(
    "rewritten", ["LicenseRef-alpha", "Apache-2.0 AND LicenseRef-alpha"]
)
def test_the_named_half_of_a_mixed_conclusion_is_checked(eula_beside_mit, rewritten):
    assert _rewrite_alpha(eula_beside_mit, rewritten) == _disagreement(rewritten)


def test_another_component_s_licence_ref_is_not_accepted(eula_beside_mit):
    """Swapped, alpha's EULA would be published as the SDK's terms; every
    extracted text would still belong to some component."""
    other = next(
        item["licenseId"]
        for item in eula_beside_mit["spdx"]["hasExtractedLicensingInfos"]
        if "demosdk" in item["licenseId"]
    )
    rewritten = f"{other} AND MIT"

    assert _rewrite_alpha(eula_beside_mit, rewritten) == _disagreement(rewritten)


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


@pytest.fixture
def shared_text(workspace, license_data):
    """alpha and beta ship the demo SDK's proprietary terms byte for byte."""

    licensing.load_corpus(license_data)
    for key in ("alpha", "beta"):
        root = workspace.build / "_deps" / f"{key}-src"
        root.mkdir(parents=True, exist_ok=True)
        (root / "LICENSE").write_text(synth.PROPRIETARY_LICENSE, encoding="utf-8")
    synth.track(workspace.root)

    payload = synth.wheel_payload(workspace)
    for key in ("alpha", "beta"):
        payload[f"isaaccapture/{key}_notice.txt"] = synth.PROPRIETARY_LICENSE.encode()
    synth.write_wheel(workspace.wheel, payload)
    return synth.collect_and_bind(workspace)


def test_components_sharing_a_licence_text_share_one_entry(shared_text):
    """A LicenseRef identifies a text, so one text is one entry.

    Two components can carry byte-identical terms -- two builds of one SDK, a
    vendor licence applied across products -- and minting an id each duplicates
    the whole text in the document.
    """
    spdx = _embedded(shared_text["wheel"])
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


def test_a_shared_id_is_held_to_the_text_it_names(shared_text):
    """gamma's Zlib text is some component's terms, so only the id catches it."""
    member = f"{DIST_INFO}/sboms/{WHEEL_NAME}.spdx.json"
    payload = synth.members(shared_text["wheel"])
    spdx = json.loads(payload[member])
    entry = next(
        item
        for item in spdx["hasExtractedLicensingInfos"]
        if item["licenseId"].startswith("LicenseRef-shared-text-")
    )
    entry["extractedText"] = synth.spdx_text("Zlib")
    payload[member] = json.dumps(spdx).encode()
    synth.write_wheel(shared_text["wheel"], payload)

    failures = validate_module.check(
        shared_text["wheel"], build_evidence=shared_text["evidence_path"]
    )

    assert failures == [
        f"{entry['licenseId']}: the text it carries is not the one its id names"
    ]


def test_a_requirement_this_run_also_built_is_not_an_unknown(
    workspace, license_data, tmp_path
):
    """A transition wheel requires the wheel beside it; that is not an unknown.

    Left as an ordinary requirement it reads as something an installer resolves,
    with version, supplier and licence unasserted -- for a distribution this same
    build produced. Stated from the sibling's own metadata rather than by a
    digest-bound reference to its document: the two wheels require each other, so
    only one direction could ever carry such a reference, and a `py3-none-any`
    wheel pinning one matrix entry's sibling stops being universal.
    """
    from isaaccapture_sbom import cli

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
                    "Author: Example Org",
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

    def sibling_row(document, name):
        spdx = json.loads((out / document).read_text())
        row = next(
            package
            for package in spdx["packages"]
            if package["name"] == name
            and package["SPDXID"].startswith("SPDXRef-Package-pypi-")
        )
        return spdx, row

    # Both directions state the same kind of thing, and neither pins a document.
    forward, isaaccapture = sibling_row(
        "isaacteleop-0.4.0-py3-none-any.spdx.json", "isaaccapture"
    )
    back, isaacteleop = sibling_row(
        f"{WHEEL_NAME.removesuffix('.whl')}.spdx.json", "isaacteleop"
    )
    for spdx, row in ((forward, isaaccapture), (back, isaacteleop)):
        assert "externalDocumentRefs" not in spdx
        assert row["versionInfo"] != "NOASSERTION"
        assert row["licenseDeclared"] == "Apache-2.0"
        assert row["supplier"] != "NOASSERTION"
        assert "Built by this same build" in row["comment"]

    assert validate_module.check(companion) == []


def test_a_wheel_shipping_no_source_still_names_its_licensor(workspace, license_data):
    """The declared licence's reference text names nobody.

    A metadata-only wheel ships no file carrying a notice, so without this the
    SPDX package states NOASSERTION and the notices file says nothing about the
    distribution -- a recipient holding only the wheel cannot tell who licensed
    it to them, for a wheel that declares one.
    """
    thin = workspace.wheel.parent / "thin-0.4.0-py3-none-any.whl"
    synth.write_wheel(
        thin,
        {
            "thin-0.4.0.dist-info/METADATA": "\n".join(
                [
                    "Metadata-Version: 2.4",
                    "Name: thin",
                    "Version: 0.4.0",
                    "Author: Example Org",
                    "License-Expression: Apache-2.0",
                    "",
                ]
            ).encode(),
            "thin-0.4.0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: setuptools\nRoot-Is-Purelib: true\n"
            ),
        },
    )
    out = workspace.root / "sbom"
    build_module.build(workspace.root, workspace.build, thin, out, license_data)

    with zipfile.ZipFile(thin) as archive:
        spdx = json.loads(
            archive.read(
                "thin-0.4.0.dist-info/sboms/thin-0.4.0-py3-none-any.whl.spdx.json"
            )
        )
        notices = archive.read(
            "thin-0.4.0.dist-info/licenses/THIRD-PARTY-NOTICES.md"
        ).decode()
        members = set(archive.namelist())

    assert _wheel_package(spdx)["copyrightText"] != "NOASSERTION"
    assert "Copyright" in notices
    assert "Example Org" in notices
    # The SPDX spelling of a supplier is not prose.
    assert "distributed by Organization:" not in notices
    # The only text it ships names no holder, which is why the above matters.
    assert "thin-0.4.0.dist-info/licenses/LICENSES/Apache-2.0.txt" in members


_IDSTRING = re.compile(r"SPDXRef-[A-Za-z0-9.-]+")


def _file_ids_of(names: list[str], version: str = "0.4.0") -> dict[str, str]:
    """Each file's id in a document over a wheel holding only these members."""
    entries = tuple(
        wheelfile.Entry(name=name, size=0, sha256="0" * 64, sha1="0" * 40)
        for name in names
    )
    spdx = document_module.build_document(
        {
            "supplier": "Organization: Example Org",
            "homepage": "https://example.invalid",
            "license": "Apache-2.0",
            "document_namespace": "https://example.invalid/spdx",
        },
        {"license_list_version": "3.27"},
        wheelfile.WheelInfo(
            path=Path(f"isaaccapture-{version}-py3-none-any.whl"),
            distribution="isaaccapture",
            version=version,
            dist_info=f"isaaccapture-{version}.dist-info",
            entries=entries,
        ),
        inventory_module.Inventory(
            attributions={},
            components_present={},
            component_roles={},
            external_runtime={},
            requires_dist=[],
        ),
        [],
    )
    return {
        item["fileName"].removeprefix("./"): item["SPDXID"] for item in spdx["files"]
    }


def test_a_file_id_is_its_sanitised_path_and_a_digest_of_it(built):
    spdx = _embedded(built["wheel"])

    for item in spdx["files"]:
        assert item["SPDXID"] == document_module.file_id(
            item["fileName"].removeprefix("./"), DIST_INFO
        )
    # Pinned: a change to what is hashed renames every file in every document.
    assert (
        document_module.file_id(
            "isaaccapture/cloudxr/oob_teleop_progress.py", DIST_INFO
        )
        == "SPDXRef-File-isaaccapture-cloudxr-oob-teleop-progress.py-f80fee5e"
    )
    assert (
        document_module.file_id(f"{DIST_INFO}/METADATA", DIST_INFO)
        == "SPDXRef-File-dist-info-METADATA-ad68becb"
    )


def test_adding_a_file_renumbers_no_other_file():
    before = _file_ids_of(["a/one.py", "b/three.py", "c/four.so"])
    after = _file_ids_of(["a/one.py", "a/two.py", "b/three.py", "c/four.so"])

    assert {name: after[name] for name in before} == before


@pytest.mark.parametrize(("old", "new"), [("1.7+local", "1.8"), ("1!1.7", "1!1.8")])
def test_a_version_bump_renames_no_file(old, new):
    def members(version):
        return [
            "isaaccapture/cloudxr/oob_teleop_progress.py",
            f"isaaccapture-{version}.dist-info/METADATA",
            f"isaaccapture-{version}.dist-info/licenses/third-party/fmt/LICENSE",
            f"isaaccapture-{version}.data/scripts/isaaccapture-tool",
        ]

    before = _file_ids_of(members(old), version=old)
    after = _file_ids_of(members(new), version=new)

    assert list(before.values()) == list(after.values())


@pytest.mark.parametrize(
    "path",
    [
        f"sub/{DIST_INFO}/METADATA",
        f"{DIST_INFO}X/f",
        f"{DIST_INFO.removesuffix('.dist-info')}.dataX/f",
    ],
)
def test_only_the_versioned_directories_are_read_as_fixed(path):
    assert document_module.file_id(path, DIST_INFO).startswith(
        f"SPDXRef-File-{licensing.spdx_safe(path)}-"
    )


def test_paths_that_sanitise_alike_get_distinct_ids():
    ids = _file_ids_of(["pkg/a_b.py", "pkg/a-b.py", "pkg/a b.py", "pkg/a__b.py"])

    assert len(set(ids.values())) == len(ids)


def test_a_path_with_no_safe_character_still_gets_a_valid_id():
    assert _IDSTRING.fullmatch(document_module.file_id("データ/名前", DIST_INFO))


def test_a_top_level_dist_info_is_not_the_versioned_one():
    ids = _file_ids_of(["dist-info/METADATA", "isaaccapture-0.4.0.dist-info/METADATA"])

    assert len(set(ids.values())) == 2


def test_two_files_one_id_stops_the_build(monkeypatch):
    monkeypatch.setattr(document_module, "file_id", lambda path, dist_info: "x")

    with pytest.raises(
        ValueError,
        match=r"isaaccapture-0\.4\.0-py3-none-any\.whl: a\.py and b\.py "
        r"would both be x",
    ):
        _file_ids_of(["a.py", "b.py"])

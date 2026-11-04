# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""External SENSING rigs and the desktop NVDEC backend.

An external rig tails files that an operator-run transport appends to; nothing is spawned. These tests cover its
configuration, YAML, inspection and rig lifecycle, the reader's attach-at-end mode, the desktop decoder pipeline
and availability check, and the composition that routes to the external rig. All CPU-only.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from capture_server import protocol as p
from capture_server.acquisition import ExternalSensingRig
from capture_server.acquisition import sensing as sn
from capture_server.acquisition import sensing_h264 as h264
from capture_server.acquisition import sensing_nvdec as nvdec
from capture_server.launch import _sensing_composition
from capture_server.server import ServerConfig

from .h264_fixtures import encode_units

W, H = 64, 32
RED, GREEN = (220, 20, 20), (20, 220, 20)
EXAMPLE_YAML = Path(__file__).parents[2] / "capture_server" / "sensing_sipl" / "sipl_remote_station.example.yaml"


def _sensor(pipeline_id=0, spool="/var/tmp/s0.h264", **over) -> sn.SensingExternalSensor:
    return sn.SensingExternalSensor(
        **{
            "module_name": f"S{pipeline_id}",
            "pipeline_id": pipeline_id,
            "width": W,
            "height": H,
            "fps": 30.0,
            "spool": spool,
            **over,
        }
    )


def _external_config(*sensors, **over) -> sn.SensingRigConfig:
    return sn.SensingRigConfig(
        expected_backend="MOCK", external=sensors or (_sensor(0), _sensor(1, "/var/tmp/s1.h264")), **over
    )


# --- desktop decoder backend ---


@pytest.fixture
def fresh_probe():
    nvdec.hardware_decoder_unavailable_reason.cache_clear()
    yield
    nvdec.hardware_decoder_unavailable_reason.cache_clear()


def test_desktop_pipeline_uses_nvcodec_and_no_jetson_only_properties():
    command = nvdec.pipeline_command(backend="desktop")
    assert "nvh264dec" in command and "videoconvert" in command and "h264parse" in command
    assert "nvv4l2decoder" not in command and "nvvidconv" not in command
    joined = " ".join(command)
    assert "enable-max-performance" not in joined and "disable-dpb" not in joined
    assert command[0] == "gst-launch-1.0" and "format=I420" in joined


def test_desktop_pipeline_keeps_a_leaky_newest_frame_queue():
    joined = " ".join(nvdec.pipeline_command(backend="desktop"))
    assert "leaky=downstream" in joined and "max-size-buffers=1" in joined


def test_jetson_pipeline_is_the_default_backend():
    assert nvdec.pipeline_command() == nvdec.pipeline_command(backend="jetson")
    assert "nvv4l2decoder" in nvdec.pipeline_command()


def test_unknown_pipeline_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be one of"):
        nvdec.pipeline_command(backend="tpu")


def test_unknown_availability_backend_is_rejected(fresh_probe):
    with pytest.raises(ValueError, match="backend must be one of"):
        nvdec.hardware_decoder_unavailable_reason("tpu")


def test_hardware_decoder_builds_the_desktop_pipeline_for_its_backend():
    decoder = nvdec.HardwareDecoder(W, H, backend="desktop")
    assert decoder._command == nvdec.pipeline_command(backend="desktop")
    assert nvdec.HardwareDecoder(W, H)._command == nvdec.pipeline_command()


def _desktop_environment(monkeypatch, *, tools=("gst-launch-1.0", "gst-inspect-1.0"), missing=()):
    """Fake a machine with no Jetson device node; fail the test if anything probes one."""

    def exists(path):
        raise AssertionError(f"the desktop check probed {path}")

    queried = []

    def run(args, **kw):
        queried.append(args[-1])
        return subprocess.CompletedProcess(args, 1 if args[-1] in missing else 0)

    monkeypatch.setattr(nvdec.os.path, "exists", exists)
    monkeypatch.setattr(nvdec.shutil, "which", lambda t: f"/usr/bin/{t}" if t in tools else None)
    monkeypatch.setattr(nvdec.subprocess, "run", run)
    return queried


def test_desktop_availability_does_not_probe_the_jetson_device_node(monkeypatch, fresh_probe):
    queried = _desktop_environment(monkeypatch)
    assert nvdec.hardware_decoder_unavailable_reason("desktop") is None
    assert queried == ["h264parse", "nvh264dec", "videoconvert"]


def test_desktop_availability_reports_missing_tools_and_elements(monkeypatch, fresh_probe):
    _desktop_environment(monkeypatch, tools=("gst-launch-1.0",))
    assert "gst-inspect-1.0 is not installed" in nvdec.hardware_decoder_unavailable_reason("desktop")
    nvdec.hardware_decoder_unavailable_reason.cache_clear()
    _desktop_environment(monkeypatch, missing=("nvh264dec",))
    assert "nvh264dec is not installed" in nvdec.hardware_decoder_unavailable_reason("desktop")


def test_desktop_availability_reports_a_probe_that_cannot_run(monkeypatch, fresh_probe):
    _desktop_environment(monkeypatch)

    def run(*a, **k):
        raise OSError("no exec")

    monkeypatch.setattr(nvdec.subprocess, "run", run)
    assert "could not query GStreamer for h264parse" in nvdec.hardware_decoder_unavailable_reason("desktop")


def test_availability_is_cached_per_backend(monkeypatch, fresh_probe):
    _desktop_environment(monkeypatch)
    assert nvdec.hardware_decoder_unavailable_reason("desktop") is None
    # The Jetson answer is separate: here the node check (which the desktop environment forbids) must run.
    with pytest.raises(AssertionError, match="probed"):
        nvdec.hardware_decoder_unavailable_reason("jetson")


def test_reader_rejects_an_unknown_hardware_backend(tmp_path):
    with pytest.raises(ValueError, match="hardware_backend must be one of"):
        h264.SensingStreamReader(str(tmp_path / "s"), width=W, height=H, hardware_backend="tpu")


def test_reader_asks_the_availability_check_about_its_own_backend(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(h264, "hardware_decoder_unavailable_reason", lambda backend: asked.append(backend) or "no")
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(encode_units(W, H, [RED, GREEN])))
    reader = h264.SensingStreamReader(str(path), width=W, height=H, decoder="auto", hardware_backend="desktop")
    reader.connect(1.0)
    reader.read_latest(0.5)
    reader.close()
    assert asked and set(asked) == {"desktop"}


def test_rig_config_hardware_backend_default_and_validation():
    assert _external_config().hardware_backend == "jetson"
    assert _external_config(hardware_backend="desktop").hardware_backend == "desktop"
    with pytest.raises(ValueError, match="hardware_backend must be one of"):
        _external_config(hardware_backend="tpu")


# --- SensingExternalSensor and the external SensingRigConfig ---


@pytest.mark.parametrize(
    "over",
    [
        dict(module_name=""),
        dict(spool=""),
        dict(pipeline_id=-1),
        dict(pipeline_id=True),
        dict(pipeline_id=1.0),
        dict(width=0),
        dict(height=-2),
        dict(fps=0),
        dict(fps=float("inf")),
    ],
)
def test_external_sensor_rejects_bad_values(over):
    with pytest.raises(ValueError):
        _sensor(**over)


def test_external_rig_needs_no_producer_fields():
    config = _external_config()
    assert config.executable == "" and config.link_masks == () and len(config.external) == 2


def test_external_rig_rejects_duplicate_pipeline_ids_and_spools():
    with pytest.raises(ValueError, match="distinct pipeline_id and spool"):
        _external_config(_sensor(0, "/a"), _sensor(0, "/b"))
    with pytest.raises(ValueError, match="distinct pipeline_id and spool"):
        _external_config(_sensor(0, "/a"), _sensor(1, "/a"))


def test_external_rig_has_no_producer_arguments():
    with pytest.raises(ValueError, match="no producer of its own"):
        _external_config().base_args()


def test_local_rig_still_requires_its_producer_fields():
    with pytest.raises(ValueError, match="needs executable"):
        sn.SensingRigConfig(expected_backend="MOCK")


# --- YAML ---


def _write_station(tmp_path, sensors: str, rig_extra: str = "") -> str:
    path = tmp_path / "station.yaml"
    path.write_text(
        f"sensing_rig:\n  expected_backend: MOCK\n  hardware_backend: desktop\n{rig_extra}"
        f"  external_sensors:\n{sensors}"
        "cameras:\n  left: {type: SENSING_COLOR, module_name: S0}\n"
    )
    return str(path)


GOOD_SENSOR = "    - {module_name: S0, pipeline_id: 0, width: 64, height: 32, fps: 30, spool: /var/tmp/s0.h264}\n"


def test_station_yaml_loads_external_sensors_and_backend(tmp_path):
    rig, cams = sn.load_sensing_station(_write_station(tmp_path, GOOD_SENSOR))
    assert rig.hardware_backend == "desktop"
    assert rig.external == (_sensor(0, "/var/tmp/s0.h264", fps=30),)
    assert cams["left"].module_name == "S0"


def test_station_yaml_without_link_masks_loads_for_an_external_rig(tmp_path):
    rig, _ = sn.load_sensing_station(_write_station(tmp_path, GOOD_SENSOR))
    assert rig.link_masks == () and rig.executable == ""


@pytest.mark.parametrize(
    "sensors,msg",
    [
        ("    - {module_name: S0, pipeline_id: 0, width: 64, height: 32, fps: 30}\n", "exactly the keys"),
        (
            "    - {module_name: S0, pipeline_id: 0, width: 64, height: 32, fps: 30, spool: /s, extra: 1}\n",
            "exactly the keys",
        ),
        ("    - not-a-mapping\n", "exactly the keys"),
        (
            "    - {module_name: S0, pipeline_id: '0', width: 64, height: 32, fps: 30, spool: /s}\n",
            "pipeline_id must be int",
        ),
        (
            "    - {module_name: S0, pipeline_id: 0, width: 0, height: 32, fps: 30, spool: /s}\n",
            "positive size",
        ),
        (
            GOOD_SENSOR + "    - {module_name: S1, pipeline_id: 0, width: 64, height: 32, fps: 30, spool: /t}\n",
            "distinct pipeline_id",
        ),
    ],
)
def test_station_yaml_rejects_bad_external_sensors(tmp_path, sensors, msg):
    with pytest.raises(ValueError, match=msg):
        sn.load_sensing_station(_write_station(tmp_path, sensors))


def test_station_yaml_rejects_external_sensors_that_are_not_a_list(tmp_path):
    path = tmp_path / "station.yaml"
    path.write_text(
        "sensing_rig:\n  expected_backend: MOCK\n  external_sensors: {a: b}\n"
        "cameras:\n  left: {type: SENSING_COLOR, module_name: S0}\n"
    )
    with pytest.raises(ValueError, match="must be a list"):
        sn.load_sensing_station(str(path))


def test_station_yaml_rejects_unknown_rig_keys_and_backends(tmp_path):
    with pytest.raises(ValueError):
        sn.load_sensing_station(_write_station(tmp_path, GOOD_SENSOR, rig_extra="  bogus: 1\n"))
    with pytest.raises(ValueError, match="hardware_backend"):
        sn.load_sensing_station(_write_station(tmp_path, GOOD_SENSOR, rig_extra="  hardware_backend: tpu\n"))


def test_the_shipped_example_station_file_parses():
    rig, cams = sn.load_sensing_station(str(EXAMPLE_YAML))
    assert rig.hardware_backend == "desktop" and rig.decoder == "hardware" and rig.expected_backend == "JETSON"
    assert [s.pipeline_id for s in rig.external] == [0, 1]
    assert {c.module_name for c in cams.values()} == {s.module_name for s in rig.external}


# --- inspection ---


def test_inspect_rig_for_an_external_rig_runs_nothing():
    def run(*a, **k):
        raise AssertionError("an external rig must not run a producer")

    description = sn.inspect_rig(_external_config(), run=run)
    assert description.sensors == (
        sn.SensingSensor("S0", 0, W, H, 30.0),
        sn.SensingSensor("S1", 1, W, H, 30.0),
    )


# --- ExternalSensingRig ---


def test_external_rig_selects_only_the_requested_spools():
    rig = ExternalSensingRig(_external_config(), (1,))
    assert rig.stream_path(1) == "/var/tmp/s1.h264"
    with pytest.raises(KeyError):
        rig.stream_path(0)


def test_external_rig_rejects_a_pipeline_without_a_configured_sensor():
    with pytest.raises(ValueError, match=r"pipeline id\(s\) \[7\]"):
        ExternalSensingRig(_external_config(), (0, 7))


def _rig_for(*paths) -> ExternalSensingRig:
    sensors = tuple(_sensor(i, str(path)) for i, path in enumerate(paths))
    return ExternalSensingRig(_external_config(*sensors), tuple(range(len(paths))))


def test_start_with_a_regular_file_records_no_error(tmp_path):
    spool = tmp_path / "s.h264"
    spool.write_bytes(b"")
    rig = _rig_for(spool)
    asyncio.run(rig.start())
    assert rig.start_error is None and rig.stream_path(0) == str(spool)


def test_start_reports_a_missing_spool_without_creating_it(tmp_path):
    spool = tmp_path / "missing.h264"
    rig = _rig_for(spool)
    asyncio.run(rig.start())
    assert "does not exist" in rig.start_error and str(spool) in rig.start_error
    assert not spool.exists()


def test_start_reports_a_non_regular_spool(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    rig = _rig_for(fifo, tmp_path)
    asyncio.run(rig.start())
    assert rig.start_error.count("is not a regular file") == 2


def test_start_reports_an_unreadable_path(tmp_path):
    rig = _rig_for(tmp_path / "file" / "child")
    (tmp_path / "file").write_bytes(b"")  # a path component that is a file: ENOTDIR, not FileNotFoundError
    asyncio.run(rig.start())
    assert rig.start_error is not None and "does not exist" not in rig.start_error


def test_start_lists_every_problem_at_once(tmp_path):
    good = tmp_path / "good"
    good.write_bytes(b"")
    rig = _rig_for(good, tmp_path / "gone")
    asyncio.run(rig.start())
    assert "gone" in rig.start_error and str(good) not in rig.start_error


def test_external_rig_owns_no_process(tmp_path):
    rig = _rig_for(tmp_path / "s")
    assert rig.exit_code() is None and rig.cleanup_ok is True
    assert asyncio.run(rig.close()) is True
    assert ExternalSensingRig.attach_at_end is True and sn.SensingRig.attach_at_end is False


def test_diagnostics_point_at_the_operator_and_keep_the_first_abort_reason(tmp_path):
    rig = _rig_for(tmp_path / "s")
    assert "run by the operator" in rig.diagnostics()
    rig.abort("stalled")
    rig.abort("later")
    assert rig.diagnostics().startswith("stalled; ")


# --- the reader's start_at_end ---


def test_start_at_end_skips_existing_bytes_and_starts_reclamation_there(tmp_path):
    old = b"".join(encode_units(W, H, [RED, RED]))
    path = tmp_path / "s.h264"
    path.write_bytes(old)
    reader = h264.SensingStreamReader(str(path), width=W, height=H, start_at_end=True)
    reader.connect(1.0)
    assert reader._offset == reader._punched == len(old)
    # Nothing new yet: the earlier session's RED frames must never be delivered.
    assert reader.read_latest(0.1) is None
    with open(path, "ab") as f:
        f.write(b"".join(encode_units(W, H, [GREEN, GREEN])))
    rgb, _ = reader.read_latest(2.0)
    assert abs(int(rgb[H // 2, W // 2][1]) - GREEN[1]) <= 14 and int(rgb[H // 2, W // 2][0]) < 100
    reader.close()


def test_default_reader_reads_existing_bytes_from_the_start(tmp_path):
    path = tmp_path / "s.h264"
    path.write_bytes(b"".join(encode_units(W, H, [RED, RED])))
    reader = h264.SensingStreamReader(str(path), width=W, height=H)
    reader.connect(1.0)
    assert reader._offset == 0
    assert reader.read_latest(1.0) is not None
    reader.close()


# --- launch routing ---


def _camera(name="left") -> p.CameraStreamConfig:
    return p.CameraStreamConfig(1, name, p.CameraType.SENSING_COLOR, True, W, H, 30.0, 80)


def test_launch_routes_an_external_config_to_the_external_rig():
    config = _external_config(hardware_backend="desktop")
    cfg = ServerConfig(
        mock=True,
        cameras={"left": sn.SensingStationCamera("S0")},
        sensing_rig=config,
        sensing_description=sn.inspect_rig(config),
    )
    rig, adapters = _sensing_composition(cfg, (_camera(),))
    assert isinstance(rig, ExternalSensingRig) and rig.stream_path(0) == "/var/tmp/s0.h264"
    assert adapters[1]._hardware_backend == "desktop"


def test_launch_still_routes_a_local_config_to_the_spawning_rig():
    local = sn.SensingRigConfig(
        executable="/x", expected_backend="MOCK", platform_config="/p", config_name="C", link_masks=("0x0",)
    )
    cfg = ServerConfig(
        mock=True,
        cameras={"left": sn.SensingStationCamera("L")},
        sensing_rig=local,
        sensing_description=sn.SensingDescription((sn.SensingSensor("L", 0, W, H, 30.0),)),
    )
    rig, _ = _sensing_composition(cfg, (_camera(),))
    assert type(rig) is sn.SensingRig


def test_adapter_attaches_at_end_only_for_an_external_rig(tmp_path, monkeypatch):
    seen = []

    class Spy:
        """Stands in for the reader and records how the adapter built it."""

        def __init__(self, *a, **kw):
            seen.append(kw)
            raise RuntimeError("stop after construction")

    monkeypatch.setattr(sn, "SensingStreamReader", Spy)
    sensor = sn.SensingSensor("S0", 0, W, H, 30.0)
    external = ExternalSensingRig(_external_config(), (0,))
    local_rig = type("R", (), {"attach_at_end": False, "start_error": None, "stream_path": lambda self, i: "/x"})()
    for rig in (external, local_rig):
        adapter = sn.SensingAdapter(
            rig=rig, sensor=sensor, camera=_camera(), hardware_backend="desktop", startup_timeout_s=0.1
        )
        with pytest.raises(Exception):
            asyncio.run(adapter.enter())
    assert [kw["start_at_end"] for kw in seen] == [True, False]
    assert all(kw["hardware_backend"] == "desktop" for kw in seen)

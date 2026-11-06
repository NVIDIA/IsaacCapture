# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Human calibration is robot-independent, side-specific, and wrap-safe."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml

from isaaccapture_examples.realhand_ffg_glove import (
    realhand_ffg_glove_calibration as _CALIBRATION,
)

_PATH = Path(_CALIBRATION.__file__)


@pytest.mark.parametrize("ports", (None, "/dev/ttyUSB0,/dev/ttyUSB1"))
def test_raw_calibration_forwards_verified_ports_without_loading_saved_mapping(
    ports, tmp_path, monkeypatch
):
    arguments = [
        str(_PATH),
        "--output",
        str(tmp_path / "human.yml"),
        "--plugin-path",
        "install/plugins",
    ]
    if ports:
        arguments.append(f"--realhand-ffg-glove-ports={ports}")
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        _CALIBRATION.CloudXRLauncher, "launch_context", lambda _: nullcontext()
    )
    captured = []

    class StopBeforeOpenXR(Exception):
        pass

    def capture(config):
        captured.append(config)
        raise StopBeforeOpenXR

    monkeypatch.setattr(_CALIBRATION, "TeleopSession", capture)
    with pytest.raises(StopBeforeOpenXR):
        _CALIBRATION.main()
    expected = ["--calibration="]
    if ports:
        expected.append(f"--ports={ports}")
    assert captured[0].plugins[0].plugin_args == expected


def test_five_general_poses_do_not_select_robot_or_contact_targets():
    keys = [key for _, key in _CALIBRATION._POSES]
    assert keys == ["open", "fist", "thumb_curl", "thumb_opposition", "spread"]
    assert not any(
        "touch" in label or "pinch" in label for label, _ in _CALIBRATION._POSES
    )


def test_cli_captures_explicit_endpoints_repeated_contacts_and_open():
    keys = [key for _, key in _CALIBRATION._CAPTURE_POSES]
    assert keys == [
        "open",
        "fist",
        "thumb_curl",
        "thumb_opposition",
        "spread",
        "thumb_index",
        "thumb_middle",
        "thumb_ring",
        "thumb_little",
        "repeat_thumb_index",
        "repeat_thumb_middle",
        "repeat_thumb_ring",
        "repeat_thumb_little",
        "open_repeat",
    ]


def test_capture_keeps_repeated_observations_and_unwraps_median():
    poses = {}
    samples = np.tile([6.27, 0.01, 0.02], 20)[:, None] * np.ones((1, 21))
    for key in ("thumb_index", "repeat_thumb_index", "open_repeat", "spread"):
        _CALIBRATION._store_capture(poses, key, samples)
    assert poses["thumb_index"] == pytest.approx(np.full(21, 2 * np.pi + 0.01))
    observations = poses["thumb_observations"]
    assert len(observations) == 48
    assert [row["pose"] for row in observations] == ["thumb_index"] * 32 + ["open"] * 16
    assert observations[1]["values"] == samples[3].tolist()


@pytest.mark.parametrize(
    "samples", [np.zeros((29, 21)), np.zeros((30, 20)), np.full((30, 21), np.nan)]
)
def test_invalid_hold_does_not_mutate_captures(samples):
    poses = {}
    with pytest.raises(ValueError, match="30 finite"):
        _CALIBRATION._store_capture(poses, "open", samples)
    assert poses == {}


def test_disconnected_status_rejects_cached_raw_value(monkeypatch):
    monkeypatch.setattr(_CALIBRATION, "_sample_group", lambda *_: np.ones(21))
    report = SimpleNamespace(
        schema_version=1,
        report_time_ns=_CALIBRATION.time.monotonic_ns(),
        devices=[
            SimpleNamespace(
                path="/glove/realhand_ffg_glove_left",
                state=SimpleNamespace(name="DISCONNECTED"),
            )
        ],
    )
    tracker = SimpleNamespace(get_device_status_snapshot=lambda _: report)
    session = SimpleNamespace(deviceio_session=None, step=lambda: None)
    assert _CALIBRATION._current_sample(session, None, "left", tracker) is None
    with pytest.raises(RuntimeError, match="glove data lost"):
        _CALIBRATION._capture_pose(
            session, ("left",), 30, 1, hold_seconds=0.1, status_tracker=tracker
        )


def test_fit_failure_keeps_file_and_other_side_unchanged(
    human_calibration, monkeypatch
):
    from isaaccapture_examples.realhand_ffg_glove import _thumb_calibration as fitter

    before = human_calibration.read_bytes()
    captured = _CALIBRATION._load_output(human_calibration)["hands"]

    def reject(*_):
        raise ValueError("bad fit")

    monkeypatch.setattr(fitter, "fit_thumb_mapping", reject)
    with pytest.raises(ValueError, match="bad fit"):
        _CALIBRATION._fit_and_save(human_calibration, {"left": captured["left"]})
    assert human_calibration.read_bytes() == before
    assert "thumb_mapping" not in captured["left"]


def test_refit_replaces_stale_coefficients_only_on_selected_side(
    human_calibration, monkeypatch
):
    from isaaccapture_examples.realhand_ffg_glove import _thumb_calibration as fitter

    original = _CALIBRATION._load_output(human_calibration)
    original["hands"]["left"]["thumb_mapping"] = {"stale": True}
    human_calibration.write_text(yaml.safe_dump(original))
    captured = {
        key: value
        for key, value in original["hands"]["left"].items()
        if key != "thumb_mapping"
    }
    mapping = {"format": fitter.FORMAT, "input_min": [-0.2] * 4, "input_max": [0.2] * 4}
    monkeypatch.setattr(fitter, "fit_thumb_mapping", lambda poses, side: mapping)
    _CALIBRATION._fit_and_save(human_calibration, {"left": captured})
    saved = _CALIBRATION._load_output(human_calibration)
    assert saved["hands"]["left"]["thumb_mapping"] == mapping
    assert saved["hands"]["right"] == original["hands"]["right"]
    assert "thumb_mapping" not in captured


def test_failed_atomic_replace_preserves_existing_calibration(
    human_calibration, monkeypatch
):
    from isaaccapture_examples.realhand_ffg_glove import _thumb_calibration as fitter

    before = human_calibration.read_bytes()
    captured = _CALIBRATION._load_output(human_calibration)["hands"]
    mapping = {"format": fitter.FORMAT, "input_min": [-0.2] * 4, "input_max": [0.2] * 4}
    monkeypatch.setattr(fitter, "fit_thumb_mapping", lambda *_: mapping)

    def fail(*_):
        raise OSError("replace failed")

    monkeypatch.setattr(_CALIBRATION.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        _CALIBRATION._fit_and_save(human_calibration, {"left": captured["left"]})
    assert human_calibration.read_bytes() == before
    assert list(human_calibration.parent.iterdir()) == [human_calibration]


def test_failed_repeated_open_check_preserves_existing_calibration(
    human_calibration, monkeypatch
):
    from isaaccapture_examples.realhand_ffg_glove import _thumb_calibration as fitter

    before = human_calibration.read_bytes()
    poses = _CALIBRATION._load_output(human_calibration)["hands"]["left"]
    poses["open_repeat"] = poses["open"]
    monkeypatch.setattr(
        fitter,
        "fit_thumb_mapping",
        lambda *_: {"fit_report": {"repeat_open_training_tip_mm": 16.0}},
    )
    with pytest.raises(ValueError, match="repeated-open"):
        _CALIBRATION._fit_and_save(human_calibration, {"left": poses})
    assert human_calibration.read_bytes() == before


def test_cli_runs_all_fourteen_holds_before_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", [str(_PATH), "--output", str(tmp_path / "human.yml")]
    )
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(
        _CALIBRATION.CloudXRLauncher, "launch_context", lambda _: nullcontext()
    )
    monkeypatch.setattr(_CALIBRATION, "TeleopSession", lambda _: nullcontext(object()))
    monkeypatch.setattr(_CALIBRATION, "_wait_for_sides", lambda *_: None)
    preparations = []
    monkeypatch.setattr(
        _CALIBRATION, "_prepare", lambda _, seconds: preparations.append(seconds)
    )
    holds = []

    def capture(session, sides, minimum, timeout, **kwargs):
        holds.append(kwargs)
        assert minimum == 120
        assert kwargs["hold_seconds"] == 6.0
        assert kwargs["status_tracker"] is not None
        assert kwargs["return_samples"]
        return {side: np.full((120, 21), len(holds) * 0.01) for side in sides}

    monkeypatch.setattr(_CALIBRATION, "_capture_pose", capture)
    results = []
    monkeypatch.setattr(
        _CALIBRATION, "_fit_and_save", lambda path, data: results.append(data)
    )
    _CALIBRATION.main()
    assert preparations == [8] * 14
    assert len(holds) == 14
    assert len(results) == 1
    for poses in results[0].values():
        assert set(poses) == {key for _, key in _CALIBRATION._CAPTURE_POSES} | {
            "thumb_observations"
        }
        assert len(poses["thumb_observations"]) == 12 * 16


def test_updating_one_side_preserves_other_measurements(human_calibration):
    data = _CALIBRATION._load_output(human_calibration)
    right = yaml.safe_dump(data["hands"]["right"])
    data["hands"]["left"]["open"][0] = 0.2
    assert yaml.safe_dump(data["hands"]["right"]) == right
    _CALIBRATION._validate_calibration(data, ("left", "right"))
    assert "hand_model" not in data


@pytest.mark.parametrize(
    "key,sensor", (("fist", 6), ("thumb_curl", 2), ("thumb_opposition", 0))
)
def test_rejects_insufficient_motion(human_calibration, key, sensor):
    data = _CALIBRATION._load_output(human_calibration)
    data["hands"]["left"][key][sensor] = data["hands"]["left"]["open"][sensor]
    with pytest.raises(ValueError, match="insufficient calibration motion"):
        _CALIBRATION._validate_calibration(data, ("left",))


def test_legacy_calibration_is_not_silently_reused(tmp_path):
    path = tmp_path / "old.yml"
    path.write_text("format: realhand-ffg-glove-raw-calibration-v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a human device calibration"):
        _CALIBRATION._load_output(path)


def test_capture_handles_periodic_sensor_phase(monkeypatch):
    readings = iter([np.full(21, 6.27), np.full(21, 0.01), np.full(21, 0.02)])
    monkeypatch.setattr(
        _CALIBRATION, "_sample_group", lambda result, side: next(readings)
    )
    monkeypatch.setattr(_CALIBRATION.time, "sleep", lambda seconds: None)

    class Session:
        def step(self):
            return None

    captured = _CALIBRATION._capture_pose(Session(), ("left",), 3, 1.0)
    assert np.asarray(captured["left"]) == pytest.approx(np.full(21, 2 * np.pi + 0.01))

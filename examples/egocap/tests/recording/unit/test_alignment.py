# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from egocap.recording import alignment
from egocap.recording.alignment import (
    FRAME_FIELD_RULES,
    AlignmentError,
    InsufficientSamplesError,
    align_episode,
    find_nearest_timestamp_indices,
    interpolate_field,
    qualified_sources,
    source_rules,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "alignment"
CAMERAS = ["head_camera", "wrist_camera"]
# The declarations a session's config.yaml carries for the golden scenarios' sources.
CONFIG = {
    "recording": {"video_frame_type": "RGBCameraFrame", "numeric_frame_type": "NumericFrame"},
    "video_sources": [{"name": name} for name in CAMERAS],
    "numeric_sources": [
        {"name": "pico_controller", "frame_type": "PicoTrackingFrame"},
        {"name": "manus_glove", "frame_type": "ManusFrame"},
    ],
}
RULES = alignment.source_rules(CONFIG)
GLOVE_FIELDS = ("left_hand_pose", "right_hand_pose", "left_sensor_count")


def load(name):
    with np.load(FIXTURES / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def inputs(data, rules=RULES):
    times = {k.split("__")[1]: v for k, v in data.items() if k.startswith("times__")}
    fields = {}
    for k, v in data.items():
        if k.startswith("field__"):
            _, source, field_name = k.split("__")
            fields.setdefault(source, {})[field_name] = v
    return times, fields


def align(data, ds=1, rules=RULES, **kwargs):
    times, fields = inputs(data)
    return align_episode("head_camera", times, CAMERAS, fields, rules, downsample=ds, **kwargs)


def ours_by_reference_key(result):
    out = {"valid_indices": result.valid_indices, "valid_timestamps": result.valid_timestamps}
    for name, source in result.sources.items():
        if source.aligned_frame_indices is not None:
            out[f"aligned_frame_indices__{name}"] = source.aligned_frame_indices
        for field_name, values in source.interpolated.items():
            out[f"interp__{name}__{field_name}"] = values
    return out


def assert_equals_reference(data, ds, rules=RULES):
    ours = ours_by_reference_key(align(data, ds, rules))
    ref = {k.split("__", 1)[1]: v for k, v in data.items() if k.startswith(f"out{ds}__")}
    ref1 = {k.split("__", 1)[1]: v for k, v in data.items() if k.startswith("out1__")}
    if ds > 1:
        # The reference's glove arrays for a stride are the matching rows of its full run (checked when generated).
        rows = np.searchsorted(np.flatnonzero(ref1["valid_indices"]), np.flatnonzero(ref["valid_indices"]))
        for field_name in GLOVE_FIELDS:
            ref[f"interp__manus_glove__{field_name}"] = ref1[f"interp__manus_glove__{field_name}"][rows]
    assert set(ours) == set(ref)
    for key, expected in ref.items():
        assert ours[key].shape == expected.shape, key
        if expected.dtype.kind in "biu":
            assert np.array_equal(ours[key], expected), key
        else:
            assert np.allclose(ours[key], expected, rtol=0, atol=1e-12), key


@pytest.mark.parametrize(
    ("scenario", "ds"),
    [
        ("01_jittered_index_offset_wrist", 1),
        ("01_jittered_index_offset_wrist", 2),
        ("01_jittered_index_offset_wrist", 3),
        ("03_tracking_late_start_early_end", 1),
        ("04_duplicate_timestamps", 1),
        ("05_irregular_tracking_with_gap", 1),
    ],
)
def test_matches_reference_alignment_outputs(scenario, ds):
    assert_equals_reference(load(scenario), ds)


def test_late_tracking_sets_the_valid_range():
    data = load("03_tracking_late_start_early_end")
    result = align(data)
    assert result.valid_range[0] == data["times__pico_controller"][0]
    assert result.valid_range[1] == data["times__pico_controller"][-1]
    assert not result.valid_indices[0]


def test_single_sample_source_matches_reference_except_where_the_reference_skips():
    # The reference logs and skips the glove's linear fields (a single sample cannot be interpolated); ours raises.
    data = load("06_single_sample_source")
    assert "out1__interp__manus_glove__left_hand_pose" not in data
    with pytest.raises(InsufficientSamplesError):
        align(data)
    without_hand_poses = {**RULES, "manus_glove": {"left_sensor_count": "nearest"}}
    assert_equals_reference(data, 1, without_hand_poses)


def test_backwards_timestamp_is_refused_where_the_reference_drops_the_sample():
    data = load("04b_backwards_timestamp")
    assert "out1__valid_indices" in data
    with pytest.raises(AlignmentError, match="backwards"):
        align(data)


def test_no_common_range_is_refused_where_the_reference_writes_empty_output():
    data = load("07_no_common_range")
    assert not data["out1__valid_indices"].any()
    times, fields = inputs(data)
    with pytest.raises(AlignmentError, match="common range"):
        align_episode("head_camera", times, ["head_camera"], fields, RULES)


def test_nearest_tie_goes_to_the_earlier_sample():
    assert find_nearest_timestamp_indices(np.array([0.0, 2.0]), np.array([1.0])).tolist() == [0]


def test_a_sample_can_serve_two_targets():
    out = find_nearest_timestamp_indices(np.array([0.0, 10.0]), np.array([0.1, 0.2, 9.9]))
    assert out.tolist() == [0, 0, 1]


def test_single_sample_source_is_nearest_to_everything():
    assert find_nearest_timestamp_indices(np.array([5.0]), np.array([1.0, 9.0])).tolist() == [0, 0]


def test_clean_drops_duplicates_and_reports_too_few_samples():
    t = np.array([0.0, 0.0, 1.0, 2.0])
    v = np.array([0.0, 99.0, 1.0, 2.0])
    out = interpolate_field("linear", t, v, np.array([0.5]))
    assert out.tolist() == [0.5]
    with pytest.raises(InsufficientSamplesError, match="1 remain"):
        interpolate_field("linear", np.array([1.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0]))


def test_interpolate_pose_stack():
    t = np.array([0.0, 1.0])
    values = np.stack([np.eye(4), np.eye(4)])
    values[1, :3, 3] = [2.0, 0.0, 0.0]
    out = interpolate_field("pose", t, values, np.array([0.5]))
    assert out.shape == (1, 4, 4)
    assert np.allclose(out[0, :3, 3], [1.0, 0.0, 0.0])
    assert out[0, 3].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_pose_with_other_shape_falls_back_to_elementwise():
    t = np.array([0.0, 1.0])
    values = np.array([[0.0, 0.0], [2.0, 4.0]])
    assert interpolate_field("pose", t, values, np.array([0.5])).tolist() == [[1.0, 2.0]]


def test_linear_on_scalars_vectors_and_hand_stacks():
    t = np.array([0.0, 2.0])
    targets = np.array([1.0])
    assert interpolate_field("linear", t, np.array([0.0, 4.0]), targets).tolist() == [2.0]
    assert interpolate_field("linear", t, np.array([[0.0, 1.0], [2.0, 3.0]]), targets).tolist() == [[1.0, 2.0]]
    hands = np.stack([np.zeros((25, 4, 4)), np.full((25, 4, 4), 2.0)])
    assert np.allclose(interpolate_field("linear", t, hands, targets), np.ones((1, 25, 4, 4)))


def test_nearest_rule_takes_the_nearest_sample_value():
    out = interpolate_field("nearest", np.array([0.0, 1.0, 2.0]), np.array([10, 11, 12]), np.array([0.9, 2.0]))
    assert out.tolist() == [11, 12]


def test_raw_rule_returns_none():
    assert interpolate_field("raw", np.array([0.0, 1.0]), np.array([1, 2]), np.array([0.5])) is None


def test_unknown_rule_is_an_error():
    with pytest.raises(AlignmentError, match="unknown field rule"):
        interpolate_field("cubic", np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([0.5]))


def simple(index=None, other=None, downsample=1, rules=None, fields=None, **kwargs):
    times = {
        "cam": np.arange(10.0) if index is None else index,
        "arm": np.arange(-2.0, 12.0, 0.5) if other is None else other,
    }
    arm_fields = {"x": times["arm"] * 2.0}
    return align_episode(
        "cam",
        times,
        ["cam"],
        {"arm": arm_fields} if fields is None else fields,
        {"arm": {"x": "linear"}} if rules is None else rules,
        downsample=downsample,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"index": None, "downsample": 0}, "at least 1"),
        ({"other": np.array([])}, "non-empty"),
        ({"other": np.array([0.0, np.nan])}, "finite"),
        ({"other": np.array([0.0, np.inf])}, "finite"),
        ({"other": np.zeros((2, 2))}, "non-empty"),
        ({"other": np.array([2.0, 1.0])}, "backwards"),
    ],
)
def test_bad_input_is_refused(kwargs, message):
    with pytest.raises(AlignmentError, match=message):
        simple(**kwargs)


def test_missing_index_source_is_refused():
    with pytest.raises(AlignmentError, match="no times"):
        align_episode("cam", {"arm": np.arange(3.0)}, [], {}, {})


def test_empty_valid_range_is_an_error_not_an_empty_success():
    with pytest.raises(AlignmentError, match="common range"):
        simple(other=np.arange(20.0, 25.0))


def test_downsample_counts_from_the_first_frame():
    # Frames 0 and 1 are outside the arm's range, so the stride keeps 4, 6, 8 (even positions), not 2, 4, 6.
    result = simple(other=np.arange(3.5, 20.0), downsample=2)
    assert result.valid_timestamps.tolist() == [4.0, 6.0, 8.0]


def test_unrecorded_and_raw_rules_add_nothing():
    result = simple(rules={"arm": {"x": "raw", "missing": "linear"}, "ghost": {"y": "linear"}})
    assert result.sources["arm"].interpolated == {}


def test_source_without_recorded_fields_adds_nothing():
    result = simple(fields={}, rules={"arm": {"x": "linear"}})
    assert result.sources["arm"].interpolated == {}


def test_index_source_fields_are_only_filtered():
    times = {"cam": np.arange(10.0), "arm": np.arange(2.0, 8.0)}
    values = np.arange(10) * 3
    result = align_episode("cam", times, ["cam"], {"cam": {"v": values}}, {"cam": {"v": "linear"}})
    assert result.sources["cam"].interpolated["v"].tolist() == [6, 9, 12, 15, 18, 21]


def test_camera_index_source_frames_are_the_valid_positions():
    result = simple()
    assert np.array_equal(result.sources["cam"].aligned_frame_indices, np.flatnonzero(result.valid_indices))
    assert result.sources["arm"].aligned_frame_indices is None


def test_valid_range_is_first_and_last_common_time():
    assert simple().valid_range == (0.0, 9.0)


def test_no_epochs_and_empty_epochs_change_nothing():
    data = load("01_jittered_index_offset_wrist")
    base = ours_by_reference_key(align(data))
    for epochs in (None, {}):
        again = ours_by_reference_key(align(data, epochs=epochs))
        assert all(np.array_equal(base[k], again[k]) for k in base)


def test_uniform_epoch_labels_leave_the_output_identical():
    data = load("01_jittered_index_offset_wrist")
    times, _ = inputs(data)
    epochs = {name: np.full(len(t), 4) for name, t in times.items()}
    base = ours_by_reference_key(align(data))
    labelled = ours_by_reference_key(align(data, epochs=epochs))
    assert set(base) == set(labelled)
    assert all(np.array_equal(base[k], labelled[k]) for k in base)


def test_changed_epoch_label_is_refused():
    labels = np.array([0] * 14 + [1] * 14)
    with pytest.raises(AlignmentError, match="restart or invalid interval"):
        simple(epochs={"arm": labels})


def test_epoch_labels_of_the_wrong_length_are_refused():
    with pytest.raises(AlignmentError, match="one per sample"):
        simple(epochs={"arm": np.zeros(3)})


def test_epoch_labels_for_an_unknown_source_are_refused():
    with pytest.raises(AlignmentError, match="one per sample"):
        simple(epochs={"ghost": np.zeros(3)})


def stream(name, meaning="sensor_capture"):
    return {"name": name, "timestamp_basis": "PUBLISHER_MONOTONIC_US", "time_meaning": meaning}


def timing(streams, physical="available: measurement time per sample"):
    return {
        "clock_mapping": {"streams": streams},
        "tracking_semantics": {"physical_alignment": physical},
    }


def test_camera_is_qualified_only_for_a_sensor_capture_stream():
    t = timing([stream("cam1"), stream("cam2", "frame_ready")])
    assert qualified_sources(t, ["cam1", "cam2", "cam3"], []) == {"cam1": True, "cam2": False, "cam3": False}


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "cam", "timestamp_basis": "PUBLISHER_MONOTONIC_US"},
        {"name": "cam"},
        {"name": "cam", "time_meaning": None},
        {"name": "other"},
        "cam",
    ],
)
def test_camera_with_missing_or_unqualified_evidence_is_false(entry):
    assert qualified_sources(timing([entry]), ["cam"], []) == {"cam": False}


def test_tracking_is_qualified_only_when_physical_alignment_is_available():
    sources = ["pico_controller", "manus_glove"]
    assert qualified_sources(timing([], "available: x"), [], sources) == dict.fromkeys(sources, True)
    assert qualified_sources(timing([], "unavailable: no time"), [], sources) == dict.fromkeys(sources, False)


@pytest.mark.parametrize(
    "t",
    [
        None,
        {},
        {"tracking_semantics": {}},
        {"tracking_semantics": {"physical_alignment": 3}},
        {"tracking_semantics": "available"},
        {"clock_mapping": {"streams": "cam"}, "tracking_semantics": []},
        {"clock_mapping": {"streams": 7}},
        {"clock_mapping": {"streams": {"name": "cam", "time_meaning": "sensor_capture"}}},
        {"clock_mapping": "cam"},
        {"clock_mapping": {}},
    ],
)
def test_malformed_timing_gives_false_and_never_raises(t):
    assert qualified_sources(t, ["cam"], ["pico_controller"]) == {"cam": False, "pico_controller": False}


def test_backends_current_tracking_text_does_not_qualify_tracking():
    livekit_recording = pytest.importorskip("egocap_web.livekit_recording")
    semantics = {"physical_alignment": livekit_recording._TRACKING_SEMANTICS["physical_alignment"]}
    assert qualified_sources({"tracking_semantics": semantics}, [], ["pico_controller"]) == {
        "pico_controller": False
    }


def test_controller_poses_are_slerped_and_buttons_stay_raw():
    rules = FRAME_FIELD_RULES["PicoTrackingFrame"]
    assert {rules[f] for f in ("head_pose", "left_pose", "right_pose")} == {"pose"}
    assert {rules[f] for f in ("left_trigger", "right_trigger")} == {"linear"}
    assert not {"a_button", "b_button", "x_button", "y_button"} & set(rules)


def test_glove_poses_are_elementwise_linear_never_slerped():
    rules = FRAME_FIELD_RULES["ManusFrame"]
    assert rules["left_hand_pose"] == rules["right_hand_pose"] == "linear"
    assert "pose" not in rules.values()


@pytest.mark.parametrize("side", ["left", "right"])
def test_glove_sensors_are_linear_and_their_counts_nearest(side):
    rules = FRAME_FIELD_RULES["ManusFrame"]
    assert [rules[f"{side}_sensor_{kind}"] for kind in ("position", "orientation", "rotation")] == ["linear"] * 3
    assert rules[f"{side}_sensor_count"] == "nearest"


def test_camera_frames_resample_a_pose_and_generic_numeric_frames_nothing():
    assert FRAME_FIELD_RULES["RGBCameraFrame"] == {"pose": "pose"}
    assert FRAME_FIELD_RULES["NumericFrame"] == {}


def test_source_rules_takes_each_sources_declared_frame_type():
    rules = source_rules(CONFIG)
    assert set(rules) == {"head_camera", "wrist_camera", "pico_controller", "manus_glove"}
    assert rules["head_camera"] == {"pose": "pose"}
    assert rules["pico_controller"] == FRAME_FIELD_RULES["PicoTrackingFrame"]
    assert rules["manus_glove"] == FRAME_FIELD_RULES["ManusFrame"]


def test_source_rules_returns_copies_so_a_caller_cannot_alter_the_table():
    source_rules(CONFIG)["pico_controller"]["head_pose"] = "raw"
    assert FRAME_FIELD_RULES["PicoTrackingFrame"]["head_pose"] == "pose"


def test_source_without_a_frame_type_takes_the_sessions_default_for_its_kind():
    config = {
        "recording": {"video_frame_type": "NumericFrame", "numeric_frame_type": "ManusFrame"},
        "video_sources": [{"name": "cam"}],
        "numeric_sources": [{"name": "glove"}, {"name": "tracker", "frame_type": "NumericFrame"}],
    }
    assert source_rules(config) == {
        "cam": {},
        "glove": FRAME_FIELD_RULES["ManusFrame"],
        "tracker": {},
    }


@pytest.mark.parametrize("recording", [None, "x", ["RGBCameraFrame"], {}])
def test_missing_or_malformed_recording_defaults_to_camera_and_numeric_frames(recording):
    config = {"video_sources": [{"name": "cam"}], "numeric_sources": [{"name": "arm"}]}
    if recording is not None:
        config["recording"] = recording
    assert source_rules(config) == {"cam": {"pose": "pose"}, "arm": {}}


def test_a_recording_section_naming_one_default_leaves_the_other():
    config = {
        "recording": {"numeric_frame_type": "ManusFrame"},
        "video_sources": [{"name": "cam"}],
        "numeric_sources": [{"name": "glove"}],
    }
    assert source_rules(config) == {"cam": {"pose": "pose"}, "glove": FRAME_FIELD_RULES["ManusFrame"]}


@pytest.mark.parametrize(
    "config", [{}, {"video_sources": None, "numeric_sources": "arm"}, {"numeric_sources": {}}]
)
def test_absent_or_malformed_sections_declare_nothing(config):
    assert source_rules(config) == {}


def test_sections_may_be_tuples_or_lists():
    as_tuples = {"video_sources": ({"name": "cam"},), "numeric_sources": ({"name": "arm"},)}
    as_lists = {"video_sources": [{"name": "cam"}], "numeric_sources": [{"name": "arm"}]}
    assert source_rules(as_tuples) == source_rules(as_lists) == {"cam": {"pose": "pose"}, "arm": {}}


def test_unknown_frame_type_is_refused_naming_the_source():
    config = {"numeric_sources": [{"name": "arm", "frame_type": "Bogus"}]}
    with pytest.raises(AlignmentError, match="arm.*Bogus"):
        source_rules(config)


def test_unknown_default_frame_type_is_refused():
    config = {"recording": {"video_frame_type": "Bogus"}, "video_sources": [{"name": "cam"}]}
    with pytest.raises(AlignmentError, match="Bogus"):
        source_rules(config)


@pytest.mark.parametrize("entry", ["cam", None, {"frame_type": "NumericFrame"}, {"name": 3}, ["name"]])
def test_an_entry_that_is_not_a_named_mapping_is_refused(entry):
    with pytest.raises(AlignmentError, match="video_sources: every source needs a name"):
        source_rules({"video_sources": [entry]})

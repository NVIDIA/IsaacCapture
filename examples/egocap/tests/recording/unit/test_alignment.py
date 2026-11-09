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
    check_continuity,
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


def by_output_key(result):
    out = {"valid_indices": result.valid_indices, "valid_timestamps": result.valid_timestamps}
    for name, source in result.sources.items():
        if source.aligned_frame_indices is not None:
            out[f"aligned_frame_indices__{name}"] = source.aligned_frame_indices
        for field_name, values in source.interpolated.items():
            out[f"interp__{name}__{field_name}"] = values
    return out


def assert_matches_expected(data, ds, rules=RULES):
    ours = by_output_key(align(data, ds, rules))
    ref = {k.split("__", 1)[1]: v for k, v in data.items() if k.startswith(f"out{ds}__")}
    ref1 = {k.split("__", 1)[1]: v for k, v in data.items() if k.startswith("out1__")}
    if ds > 1:
        # The expected glove arrays for a stride are the matching rows of the full-rate run.
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
def test_matches_expected_alignment_outputs(scenario, ds):
    assert_matches_expected(load(scenario), ds)


def test_late_tracking_sets_the_valid_range():
    data = load("03_tracking_late_start_early_end")
    result = align(data)
    assert result.valid_range[0] == data["times__pico_controller"][0]
    assert result.valid_range[1] == data["times__pico_controller"][-1]
    assert not result.valid_indices[0]


def test_single_sample_source_is_refused_for_its_linear_fields():
    # A single sample cannot be interpolated, so the glove's linear fields are refused rather than skipped.
    data = load("06_single_sample_source")
    assert "out1__interp__manus_glove__left_hand_pose" not in data
    with pytest.raises(InsufficientSamplesError):
        align(data)
    without_hand_poses = {**RULES, "manus_glove": {"left_sensor_count": "nearest"}}
    assert_matches_expected(data, 1, without_hand_poses)


def test_backwards_timestamp_is_refused():
    data = load("04b_backwards_timestamp")
    assert "out1__valid_indices" in data
    with pytest.raises(AlignmentError, match="backwards"):
        align(data)


def test_no_common_range_is_refused():
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
    base = by_output_key(align(data))
    for epochs in (None, {}):
        again = by_output_key(align(data, epochs=epochs))
        assert all(np.array_equal(base[k], again[k]) for k in base)


def test_uniform_epoch_labels_leave_the_output_identical():
    data = load("01_jittered_index_offset_wrist")
    times, _ = inputs(data)
    epochs = {name: np.full(len(t), 4) for name, t in times.items()}
    base = by_output_key(align(data))
    labelled = by_output_key(align(data, epochs=epochs))
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


def event_time(**entries):
    """A timing attribute shaped like the backend's, with per-source event-time entries."""
    return {"event_time": entries}


# What the backend records today: a sensor-capture camera qualified, a frame-ready camera and both tracking
# sources not.
REALISTIC = event_time(
    cam_capture={"meaning": "sensor_capture", "qualified": True},
    cam_ready={"meaning": "frame_ready", "qualified": False},
    pico_controller={"meaning": "pose_evaluation_time", "qualified": False},
    manus_glove={"meaning": "host_push_time", "qualified": False},
)


def test_only_sources_recorded_as_qualified_are_qualified():
    sources = ["cam_capture", "cam_ready", "pico_controller", "manus_glove", "unlisted"]
    assert qualified_sources(REALISTIC, sources) == {
        "cam_capture": True,
        "cam_ready": False,
        "pico_controller": False,
        "manus_glove": False,
        "unlisted": False,
    }


def test_the_result_has_exactly_the_requested_sources():
    assert qualified_sources(REALISTIC, []) == {}
    assert list(qualified_sources(REALISTIC, ["manus_glove", "cam_capture"])) == ["manus_glove", "cam_capture"]


@pytest.mark.parametrize("qualified", [False, None, "true", "True", 1, 1.0, [True], {}])
def test_a_qualified_value_that_is_not_literally_true_is_false(qualified):
    assert qualified_sources(event_time(cam={"qualified": qualified}), ["cam"]) == {"cam": False}


def test_an_entry_without_a_qualified_key_is_false():
    assert qualified_sources(event_time(cam={"meaning": "sensor_capture"}), ["cam"]) == {"cam": False}


@pytest.mark.parametrize("entry", [None, True, "cam", 1, [("qualified", True)]])
def test_an_entry_that_is_not_a_mapping_is_false(entry):
    assert qualified_sources(event_time(cam=entry), ["cam"]) == {"cam": False}


@pytest.mark.parametrize("value", [None, [], "cam", 7, [{"qualified": True}], True])
def test_an_event_time_that_is_not_a_mapping_gives_false(value):
    assert qualified_sources({"event_time": value}, ["cam"]) == {"cam": False}


@pytest.mark.parametrize("t", [None, {}, {"clock_mapping": {"streams": [{"name": "cam"}]}}])
def test_timing_without_event_time_gives_false(t):
    assert qualified_sources(t, ["cam"]) == {"cam": False}


def test_a_clock_mapping_or_physical_alignment_text_is_never_qualification():
    t = {
        "clock_mapping": {"streams": [{"name": "cam", "time_meaning": "sensor_capture"}]},
        "tracking_semantics": {"physical_alignment": "available: measurement time per sample"},
    }
    assert qualified_sources(t, ["cam", "pico_controller"]) == {"cam": False, "pico_controller": False}


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


def clean(scope="clock_mapping", **changes):
    return {"scope": scope, "identity": 1, "breaks": [], "known": True, **changes}


def test_only_a_clock_mapping_scope_is_accepted_as_continuity_evidence():
    assert alignment.ACCEPTED_CONTINUITY_SCOPES == ("clock_mapping",)


def test_clean_clock_mapping_evidence_for_every_source_is_accepted():
    assert check_continuity({"cam": clean(), "arm": clean()}, ["cam", "arm"]) == []


def test_evidence_for_sources_that_are_not_asked_about_is_ignored():
    assert check_continuity({"cam": clean(), "other": clean(known=False)}, ["cam"]) == []


def test_no_evidence_at_all_is_one_problem_not_a_clean_pass():
    assert check_continuity(None, ["cam", "arm"]) == ["no continuity evidence was recorded for this episode"]


def test_a_source_without_an_entry_is_refused():
    assert check_continuity({"cam": clean()}, ["cam", "arm"]) == ["arm: no continuity evidence"]


def test_an_entry_that_is_not_a_mapping_is_refused():
    assert check_continuity({"cam": [1, 2]}, ["cam"]) == ["cam: no continuity evidence"]


@pytest.mark.parametrize("known", [False, None, "yes", 1])
def test_continuity_that_is_not_exactly_known_true_is_unknown(known):
    assert check_continuity({"cam": clean(known=known)}, ["cam"]) == ["cam: continuity is unknown"]


def test_a_missing_known_flag_is_unknown():
    entry = clean()
    del entry["known"]
    assert check_continuity({"cam": entry}, ["cam"]) == ["cam: continuity is unknown"]


@pytest.mark.parametrize("scope", ["capture_session", None, "other"])
def test_a_scope_that_does_not_cover_the_source_is_refused(scope):
    # Transport continuity says the connection stayed up, not that the device behind it did not restart.
    (problem,) = check_continuity({"cam": clean(scope)}, ["cam"])
    assert problem.startswith("cam: its continuity evidence (") and f"{scope!r}" in problem


def test_a_break_is_reported_with_its_rows():
    problems = check_continuity({"cam": clean(breaks=[4, 9])}, ["cam"])
    assert problems == ["cam: has a restart inside the episode (rows [4, 9])"]


def test_every_refused_source_is_reported_at_once():
    evidence = {
        "a": clean(breaks=[2]),
        "b": clean("capture_session"),
        "c": clean(known=False),
        "ok": clean(),
    }
    problems = check_continuity(evidence, ["a", "b", "c", "ok", "missing"])
    assert [p.split(":")[0] for p in problems] == ["a", "b", "c", "missing"]


def test_missing_identity_is_incomplete():
    entry = clean()
    del entry["identity"]
    assert check_continuity({"cam": entry}, ["cam"]) == ["cam: continuity evidence is incomplete or malformed"]


@pytest.mark.parametrize("identity", [None, True, False, 1.5, [1]])
def test_an_identity_that_is_not_a_str_or_int_is_incomplete(identity):
    assert check_continuity({"cam": clean(identity=identity)}, ["cam"]) == [
        "cam: continuity evidence is incomplete or malformed"
    ]


@pytest.mark.parametrize("identity", ["epoch-a", 0, 7])
def test_str_and_int_identities_are_complete(identity):
    assert check_continuity({"cam": clean(identity=identity)}, ["cam"]) == []


def test_missing_breaks_is_incomplete():
    entry = clean()
    del entry["breaks"]
    assert check_continuity({"cam": entry}, ["cam"]) == ["cam: continuity evidence is incomplete or malformed"]


@pytest.mark.parametrize("breaks", [None, 0, (), (3,), "3", [True], [-1], [1.0], [2, "3"], [3, False]])
def test_breaks_that_are_not_a_list_of_non_negative_ints_are_incomplete(breaks):
    assert check_continuity({"cam": clean(breaks=breaks)}, ["cam"]) == [
        "cam: continuity evidence is incomplete or malformed"
    ]


def test_a_break_beyond_the_written_rows_is_malformed():
    problems = check_continuity({"cam": clean(breaks=[10])}, ["cam"], rows={"cam": 10})
    assert problems == ["cam: continuity evidence is incomplete or malformed"]


def test_a_break_at_the_last_row_is_a_restart_not_malformed():
    problems = check_continuity({"cam": clean(breaks=[9])}, ["cam"], rows={"cam": 10})
    assert problems == ["cam: has a restart inside the episode (rows [9])"]


def test_rows_for_other_sources_do_not_bound_this_one():
    assert check_continuity({"cam": clean(breaks=[])}, ["cam"], rows={"other": 1}) == []
    problems = check_continuity({"cam": clean(breaks=[50])}, ["cam"], rows={"other": 1})
    assert problems == ["cam: has a restart inside the episode (rows [50])"]

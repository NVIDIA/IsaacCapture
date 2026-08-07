# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for full-body skeleton orientation correction.

Covers:
- correct_body_orientations: recorded-capture regression, profiles, validation
- SKELETON_PROFILES and BODY_PARENT_INDICES table invariants
- FullBodyTransform node: passthrough fields, Optional propagation, aliasing

The recorded fixtures below are single frames of real Quest 3 body tracking, with
expected outputs pinned from the implementation that was validated against a
PICO 4 Ultra on hardware. They guard the tables and the frame algebra together:
a change to either moves these numbers.
"""

import pytest
import numpy as np
import numpy.testing as npt

from isaacteleop.retargeting_engine.interface import TensorGroup
from isaacteleop.retargeting_engine.interface.tensor_group import OptionalTensorGroup
from isaacteleop.retargeting_engine.tensor_types import (
    FullBodyInput,
    FullBodyInputIndex,
    NUM_BODY_JOINTS,
)
from isaacteleop.retargeting_engine.utilities import (
    FullBodyTransform,
    correct_body_orientations,
    SKELETON_PROFILES,
)
from isaacteleop.retargeting_engine.utilities.full_body_transform import (
    BODY_PARENT_INDICES,
)


# ============================================================================
# Recorded fixtures
# ============================================================================

# Quest capture '01_tpose', frame 528 of 1057: raw global orientations.
_TPOSE_RAW = np.array(
    [
        [-0.53183651, +0.43034163, -0.58225608, +0.43924237],
        [+0.43700677, +0.39538640, +0.57550669, +0.56699800],
        [-0.50029277, +0.53240138, -0.44573700, +0.51727598],
        [-0.53667873, +0.42856455, -0.56747186, +0.45418504],
        [+0.36656019, +0.46150199, +0.47797352, +0.65129936],
        [-0.57905203, +0.44552889, -0.52454924, +0.43709364],
        [-0.50233795, +0.46839966, -0.52190448, +0.50583996],
        [-0.00825935, +0.76220989, +0.09221342, +0.64067507],
        [+0.59689787, -0.09937382, +0.79528703, +0.03682808],
        [-0.48048788, +0.50199008, -0.48383748, +0.53201377],
        [-0.01007206, +0.81009059, -0.01986339, +0.58588158],
        [-0.54397046, +0.05889403, -0.83523756, +0.05482560],
        [-0.54805481, +0.41778999, -0.58447569, +0.42834055],
        [-0.66121013, +0.28245566, +0.10783485, +0.68657965],
        [+0.70309968, -0.11203927, +0.26028952, +0.65218664],
        [-0.40811863, +0.55529439, -0.44009071, +0.57568002],
        [+0.01584025, +0.01396287, +0.04556562, +0.99873815],
        [-0.99650826, +0.07576928, -0.03089592, +0.01660581],
        [+0.01507637, +0.01478446, -0.00753687, +0.99974863],
        [+0.99929838, -0.01313119, +0.03187649, -0.01463579],
        [-0.70272845, -0.05889549, +0.01398416, +0.70887834],
        [+0.73667085, -0.06165301, -0.04806288, +0.67171788],
        [-0.70272845, -0.05889549, +0.01398416, +0.70887834],
        [+0.73667085, -0.06165301, -0.04806288, +0.67171788],
    ],
    dtype=np.float32,
)

# Corrected '01_tpose', pinned from the validated implementation.
_TPOSE_EXPECTED = np.array(
    [
        [-0.12876034, -0.05906495, -0.03105445, +0.98942798],
        [-0.00200534, -0.11381048, -0.21833525, +0.96921250],
        [+0.06189095, -0.06963629, +0.15161162, +0.98403974],
        [-0.17955729, -0.05124126, -0.01909381, +0.98222652],
        [+0.10298159, +0.13233715, +0.20593741, -0.96409100],
        [-0.05382662, -0.08989232, +0.12634265, +0.98643783],
        [-0.29074884, -0.04503125, +0.03795339, +0.95498525],
        [-0.10899456, -0.04192023, +0.22378866, -0.96761641],
        [-0.08283077, +0.20242583, -0.13677865, -0.96615446],
        [-0.30633195, -0.01570987, +0.06324104, +0.94969180],
        [-0.10246462, -0.06445392, +0.28965181, -0.94944643],
        [+0.07660954, -0.21353257, +0.16236278, +0.96029847],
        [+0.44903672, +0.01645642, -0.08052031, -0.88972563],
        [-0.38415491, +0.04574109, +0.06609128, +0.91976339],
        [+0.30299349, -0.00286412, -0.06489926, -0.95077591],
        [+0.31445013, +0.02440167, -0.09153429, -0.94453541],
        [+0.00543634, -0.04561595, +0.06244006, -0.99699091],
        [-0.09223440, +0.03520401, -0.20026582, -0.97475489],
        [+0.00608601, -0.03255892, +0.10035168, -0.99440053],
        [-0.15548066, +0.02458218, -0.24942135, -0.95551581],
        [-0.00761792, +0.01263287, -0.05315160, -0.99847748],
        [-0.21470151, -0.05973986, -0.12956693, -0.96620227],
        [-0.00761792, +0.01263287, -0.05315160, -0.99847748],
        [-0.09850520, +0.03582449, +0.04539479, -0.99345490],
    ],
    dtype=np.float32,
)

# Quest capture '08_squat', frame 537 of 1074: raw global orientations.
_SQUAT_RAW = np.array(
    [
        [-0.38578749, +0.67836946, -0.30646390, +0.54503465],
        [+0.82542576, +0.10548357, +0.55127343, +0.06035842],
        [-0.06964875, +0.71764957, -0.09997173, +0.68566304],
        [-0.42274451, +0.65494572, -0.33253705, +0.53080344],
        [+0.49125870, +0.67029165, +0.33746137, +0.44214683],
        [-0.56677784, +0.44789425, -0.56302084, +0.40144882],
        [-0.55401027, +0.55136316, -0.43284314, +0.44913038],
        [-0.09950215, +0.87187821, +0.10748591, +0.46730557],
        [+0.64270401, -0.09383573, +0.74867493, +0.13271117],
        [-0.57458085, +0.52422654, -0.45121273, +0.43755051],
        [-0.02046606, +0.90725847, +0.04108122, +0.41806165],
        [+0.60159462, -0.04503143, +0.79664467, +0.03759461],
        [-0.59850830, +0.45556471, -0.53821600, +0.38022644],
        [-0.70466285, +0.19408358, +0.11842163, +0.67212956],
        [+0.69500799, +0.03423228, +0.11332996, +0.70918853],
        [-0.50358420, +0.55870782, -0.45831790, +0.47349047],
        [-0.31917208, -0.64213120, -0.00209173, +0.69698804],
        [+0.60612331, +0.07109002, -0.70873014, +0.35392418],
        [-0.16798961, -0.69712466, -0.15741882, +0.67898160],
        [+0.57068125, +0.21624594, -0.77348998, +0.17109601],
        [-0.64583874, -0.43260208, -0.38241786, +0.49950409],
        [+0.37045298, +0.49301740, -0.48491373, +0.62012668],
        [-0.64583874, -0.43260208, -0.38241786, +0.49950409],
        [+0.37045298, +0.49301740, -0.48491373, +0.62012668],
    ],
    dtype=np.float32,
)

# Corrected '08_squat', pinned from the validated implementation.
_SQUAT_EXPECTED = np.array(
    [
        [+0.25949671, +0.07714492, -0.02244805, +0.96239608],
        [+0.60071768, +0.07688039, -0.11086839, +0.78799484],
        [+0.63802375, +0.04025526, -0.02246018, +0.76863564],
        [+0.14687346, +0.08637654, -0.00515993, +0.98536321],
        [+0.05922487, -0.07788447, -0.10768010, -0.98935910],
        [-0.06905111, +0.04643306, -0.09204223, +0.99227221],
        [+0.27064204, +0.04608037, -0.11806062, +0.95430141],
        [+0.00547616, -0.17982643, -0.13642203, -0.97417734],
        [+0.03188106, +0.03538866, +0.08544687, -0.99520353],
        [+0.32220094, +0.03772070, -0.15256228, +0.93353546],
        [-0.00977366, -0.16846649, -0.17549227, -0.96991030],
        [+0.02779146, +0.04249235, +0.06321663, -0.99670743],
        [-0.23708398, -0.04162752, +0.15127496, -0.95873574],
        [+0.40566860, -0.01431182, -0.10767385, +0.90754311],
        [-0.25314995, -0.07826394, +0.12811734, -0.95570697],
        [-0.28825322, -0.04248444, +0.18971083, -0.93761130],
        [-0.36359724, +0.61881211, -0.01313126, -0.69620126],
        [+0.39491094, +0.55895949, -0.27535001, +0.67512370],
        [-0.44659233, +0.69716785, +0.06190387, -0.55738692],
        [-0.56063289, -0.59127117, +0.19869997, -0.54461683],
        [-0.56819724, +0.67127212, -0.12474435, -0.45933047],
        [-0.64393664, -0.51641437, +0.37778244, -0.41945468],
        [-0.56819724, +0.67127212, -0.12474435, -0.45933047],
        [-0.61991967, -0.56626022, +0.37500047, -0.39296771],
    ],
    dtype=np.float32,
)

_RECORDED = {
    "tpose": (_TPOSE_RAW, _TPOSE_EXPECTED),
    "squat": (_SQUAT_RAW, _SQUAT_EXPECTED),
}


# ============================================================================
# Helpers
# ============================================================================


def _identity_quats() -> np.ndarray:
    q = np.zeros((NUM_BODY_JOINTS, 4), dtype=np.float32)
    q[:, 3] = 1.0
    return q


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-joint geodesic angle between two xyzw quaternion sets, sign-agnostic."""
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))


def _make_body_input(orientations: np.ndarray = None) -> TensorGroup:
    tg = TensorGroup(FullBodyInput())
    positions = np.arange(NUM_BODY_JOINTS * 3, dtype=np.float32).reshape(
        NUM_BODY_JOINTS, 3
    )
    if orientations is None:
        orientations = _identity_quats()
    valid = np.ones(NUM_BODY_JOINTS, dtype=np.uint8)
    valid[5] = 0  # a distinguishable pattern, so passthrough is checked not assumed
    tg[FullBodyInputIndex.JOINT_POSITIONS] = positions
    tg[FullBodyInputIndex.JOINT_ORIENTATIONS] = orientations.astype(np.float32)
    tg[FullBodyInputIndex.JOINT_VALID] = valid
    return tg


# ============================================================================
# Tests: recorded-capture regression
# ============================================================================


class TestRecordedCaptures:
    @pytest.mark.parametrize("pose", sorted(_RECORDED))
    def test_matches_pinned_output(self, pose):
        raw, expected = _RECORDED[pose]
        result = correct_body_orientations(raw, "quest")
        npt.assert_allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize("pose", sorted(_RECORDED))
    def test_correction_is_not_a_no_op(self, pose):
        """Guards against a table of identities silently passing the pinned test."""
        raw, expected = _RECORDED[pose]
        assert _quat_angle_deg(raw, expected).mean() > 30.0

    @pytest.mark.parametrize("pose", sorted(_RECORDED))
    def test_output_is_unit_norm(self, pose):
        raw, _ = _RECORDED[pose]
        norms = np.linalg.norm(correct_body_orientations(raw, "quest"), axis=-1)
        npt.assert_allclose(norms, 1.0, atol=1e-9)

    def test_non_unit_input_is_normalized(self):
        """Trackers can emit slightly denormalized quaternions; output must be unit."""
        scaled = _TPOSE_RAW * 1.05
        result = correct_body_orientations(scaled, "quest")
        npt.assert_allclose(np.linalg.norm(result, axis=-1), 1.0, atol=1e-9)
        npt.assert_allclose(result, _TPOSE_EXPECTED, atol=1e-6)

    def test_distinct_poses_give_distinct_output(self):
        """The correction must depend on the input, not just the tables."""
        out_t = correct_body_orientations(_TPOSE_RAW, "quest")
        out_s = correct_body_orientations(_SQUAT_RAW, "quest")
        assert _quat_angle_deg(out_t, out_s).max() > 30.0

    def test_deterministic(self):
        a = correct_body_orientations(_TPOSE_RAW, "quest")
        b = correct_body_orientations(_TPOSE_RAW, "quest")
        npt.assert_array_equal(a, b)


# ============================================================================
# Tests: profiles
# ============================================================================


class TestProfiles:
    def test_pico_is_exact_identity(self):
        """A genuine PICO skeleton is already correct; altering it is a bug."""
        result = correct_body_orientations(_TPOSE_RAW, "pico")
        npt.assert_array_equal(result, _TPOSE_RAW.astype(np.float64))

    def test_default_profile_is_quest(self):
        npt.assert_allclose(
            correct_body_orientations(_TPOSE_RAW),
            correct_body_orientations(_TPOSE_RAW, "quest"),
        )

    @pytest.mark.parametrize(
        "profile", ["quest3", "Quest", "QUEST", "", "meta", "pico4"]
    )
    def test_unknown_profile_raises(self, profile):
        """'quest3' is the trap: it appears in deviceProfile on PICO sessions too."""
        with pytest.raises(KeyError):
            correct_body_orientations(_TPOSE_RAW, profile)

    def test_known_profiles(self):
        assert set(SKELETON_PROFILES) == {"pico", "quest"}


# ============================================================================
# Tests: input validation and purity
# ============================================================================


class TestInputHandling:
    @pytest.mark.parametrize(
        "shape", [(23, 4), (25, 4), (24, 3), (24, 5), (4, 24), (24,), (1, 24, 4)]
    )
    def test_wrong_shape_raises(self, shape):
        with pytest.raises(ValueError):
            correct_body_orientations(np.zeros(shape, dtype=np.float32), "quest")

    def test_accepts_nested_sequence(self):
        result = correct_body_orientations(_TPOSE_RAW.tolist(), "quest")
        npt.assert_allclose(result, _TPOSE_EXPECTED, atol=1e-6)

    @pytest.mark.parametrize("profile", ["quest", "pico"])
    def test_input_is_not_mutated(self, profile):
        raw = _TPOSE_RAW.copy()
        correct_body_orientations(raw, profile)
        npt.assert_array_equal(raw, _TPOSE_RAW)

    def test_output_does_not_alias_input(self):
        raw = _TPOSE_RAW.copy()
        out = correct_body_orientations(raw, "pico")
        out[0, 0] = 12345.0
        assert raw[0, 0] != 12345.0


# ============================================================================
# Tests: table invariants
# ============================================================================


class TestTables:
    @pytest.mark.parametrize("profile", sorted(SKELETON_PROFILES))
    def test_tables_are_unit_quaternions(self, profile):
        for table in SKELETON_PROFILES[profile]:
            arr = np.asarray(table, dtype=np.float64)
            assert arr.shape == (NUM_BODY_JOINTS, 4)
            npt.assert_allclose(np.linalg.norm(arr, axis=-1), 1.0, atol=1e-6)

    def test_parent_table_length(self):
        assert len(BODY_PARENT_INDICES) == NUM_BODY_JOINTS

    def test_single_root(self):
        assert BODY_PARENT_INDICES[0] == -1
        assert sum(1 for p in BODY_PARENT_INDICES if p < 0) == 1

    def test_parents_precede_children(self):
        """The global<->local conversions are single forward passes; they rely on this."""
        for joint, parent in enumerate(BODY_PARENT_INDICES):
            assert parent < joint, f"joint {joint} has parent {parent}"

    def test_no_cycles_and_all_reach_root(self):
        for joint in range(NUM_BODY_JOINTS):
            seen, cur = set(), joint
            while cur >= 0:
                assert cur not in seen, f"cycle at joint {joint}"
                seen.add(cur)
                cur = BODY_PARENT_INDICES[cur]


# ============================================================================
# Tests: FullBodyTransform node
# ============================================================================


class TestFullBodyTransform:
    def test_orientations_match_helper(self):
        node = FullBodyTransform("body_fix", profile="quest")
        result = node({"full_body": _make_body_input(_TPOSE_RAW)})
        out = np.from_dlpack(result["full_body"][FullBodyInputIndex.JOINT_ORIENTATIONS])
        npt.assert_allclose(out, _TPOSE_EXPECTED, atol=1e-6)

    def test_positions_and_validity_pass_through(self):
        node = FullBodyTransform("body_fix", profile="quest")
        inp = _make_body_input(_TPOSE_RAW)
        result = node({"full_body": inp})
        for field in (
            FullBodyInputIndex.JOINT_POSITIONS,
            FullBodyInputIndex.JOINT_VALID,
        ):
            npt.assert_array_equal(
                np.from_dlpack(result["full_body"][field]), np.from_dlpack(inp[field])
            )

    def test_pico_profile_leaves_orientations_untouched(self):
        node = FullBodyTransform("body_fix", profile="pico")
        result = node({"full_body": _make_body_input(_TPOSE_RAW)})
        out = np.from_dlpack(result["full_body"][FullBodyInputIndex.JOINT_ORIENTATIONS])
        npt.assert_array_equal(out, _TPOSE_RAW)

    def test_absent_input_propagates(self):
        node = FullBodyTransform("body_fix", profile="quest")
        result = node({"full_body": OptionalTensorGroup(FullBodyInput())})
        assert result["full_body"].is_none

    def test_present_after_absent(self):
        """Absent one call must not latch the output none on the next."""
        node = FullBodyTransform("body_fix", profile="quest")
        assert node({"full_body": OptionalTensorGroup(FullBodyInput())})[
            "full_body"
        ].is_none
        result = node({"full_body": _make_body_input(_TPOSE_RAW)})
        assert not result["full_body"].is_none
        out = np.from_dlpack(result["full_body"][FullBodyInputIndex.JOINT_ORIENTATIONS])
        npt.assert_allclose(out, _TPOSE_EXPECTED, atol=1e-6)

    def test_output_does_not_alias_input(self):
        node = FullBodyTransform("body_fix", profile="quest")
        inp = _make_body_input(_TPOSE_RAW)
        before = np.from_dlpack(inp[FullBodyInputIndex.JOINT_ORIENTATIONS]).copy()
        result = node({"full_body": inp})
        np.from_dlpack(result["full_body"][FullBodyInputIndex.JOINT_ORIENTATIONS])[
            :
        ] = 0.0
        npt.assert_array_equal(
            np.from_dlpack(inp[FullBodyInputIndex.JOINT_ORIENTATIONS]), before
        )

    def test_repeated_calls_are_stable(self):
        """In-place orientation writes must not accumulate across calls."""
        node = FullBodyTransform("body_fix", profile="quest")
        for _ in range(3):
            result = node({"full_body": _make_body_input(_TPOSE_RAW)})
            out = np.from_dlpack(
                result["full_body"][FullBodyInputIndex.JOINT_ORIENTATIONS]
            )
            npt.assert_allclose(out, _TPOSE_EXPECTED, atol=1e-6)

    def test_profile_property(self):
        assert FullBodyTransform("body_fix", profile="pico").profile == "pico"

    def test_default_profile_is_quest(self):
        assert FullBodyTransform("body_fix").profile == "quest"

    @pytest.mark.parametrize("profile", ["quest3", "Quest", "", "meta"])
    def test_unknown_profile_raises_at_construction(self, profile):
        """Fail at build time, not silently mid-session on a live robot."""
        with pytest.raises(KeyError):
            FullBodyTransform("body_fix", profile=profile)

    def test_io_spec(self):
        node = FullBodyTransform("body_fix")
        assert list(node.input_spec()) == ["full_body"]
        assert list(node.output_spec()) == ["full_body"]

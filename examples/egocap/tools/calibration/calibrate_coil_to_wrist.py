# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Find where a glove's main coil sits relative to the wrist.

Model (position only, five fingertips): ``a_t + R_At x_k = R_Y b_t + t_Y``. ``a_t``, ``R_At`` are a
fingertip sensor's pose in the main-coil frame (``manus_glove/{side}_sensor_pose``), ``b_t`` the same
fingertip in the wrist frame (``manus_glove/{side}_hand_pose``), ``x_k`` each finger's unknown tip offset in
its sensor frame and ``Y`` the coil-from-wrist transform.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# tools/calibration/ is a plain scripts directory, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as common  # noqa: E402
import record  # noqa: E402

#: What the operator does to record the take; shown by --help and before recording.
PROCEDURE = """What you need:
  glove        worn, each fingertip sensor pushed onto its fingertip, nothing in the hand
  The other glove, the controllers and the headset are not needed.

Steps, about 100 seconds. Move slowly, 3-4 seconds per motion, and keep the hand off metal.
  1. Hold the open hand still for two seconds.
  2. Bend each finger on its own, index to little, three times each.
     Then move the thumb across the palm to the base of the little finger, three times.
  3. Three times each: make a fist; bend only the knuckles; bend only the fingers.
  4. Spread the fingers and close them, three times. Wag the index, then the little finger, three times each.
  5. Thumb: touch each fingertip for a second, twice round. Draw three big circles each way.
     Lift and lower it three times.
  6. Count from one to five on your fingers and back, twice. Then tap the fingers as on a piano, twice.
  7. Hold the open hand still for two seconds, then stop.
"""

#: The fingertip rows of the 25-joint ``hand_pose`` stack, thumb to little finger.
TIP = [4, 9, 14, 19, 24]
SEG_FLOOR_MM = 3.0
#: Neighbouring frames are strongly correlated, so fits use a subset; residuals and scores use every frame.
FIT_STRIDE = 4
HELD_OUT_STRIDE = 16
HUBER_MM = 5.0
RANDOM_STARTS = 6
#: Each check as (label, comparison, threshold, unit, what to change when it fails). ``x_k`` is identified by the
#: sensor rotation and ``Y`` by the fingertip travel, both measured on kept frames.
ACCEPT = {
    "rms": ("fit error", "<", 2.5, "mm", "A fingertip sensor is loose. Push each one onto its fingertip."),
    "half_split": ("first half vs second half", "<", 1.5, "deg", "The glove moved on the hand. Put it on firmly."),
    "dropped": ("part of the take thrown out", "<", 0.50, "", "Move slower, and keep fists short."),
    "sensor_span": ("least-turned sensor: {still}", ">", 60.0, "deg", "Bend the {still} more."),
    "tip_spread": ("least-moved fingertip: {narrow}", ">", 20.0, "mm", "Move the {narrow} more."),
    "tip_offset": ("sensor distance from fingertip", "<", 30.0, "mm", "Push each sensor onto its fingertip."),
    "ambiguity": ("other answers that fit as well", "<", 1.0, "deg", "Do every step, and move every finger."),
}


def load(episode: Path, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One hand's times, usable mask, (N,5,4,4) sensor poses and (N,5,3) wrist-frame fingertip positions."""
    hand, sensor = f"{side}_hand_pose", f"{side}_sensor_pose"
    data = common.read_source(episode, "manus_glove", (hand, sensor), common.GLOVE_ROWS_HINT)
    # The recorder zeroes an unreported sensor set and an invalid joint, so a real transform ends in 1.
    reported = (data[sensor][:, :, 3, 3] == 1).all(axis=1)
    tips_valid = (data[hand][:, TIP, 3, 3] == 1).all(axis=1)
    times = data["capture_time"] - data["capture_time"][0]
    common.require_take(times[reported], f"{side} fingertip-sensor data", common.SENSOR_HINT)
    return times, reported & tips_valid, data[sensor], data[hand][:, TIP][:, :, :3, 3]


class Problem:
    """The fit in point-level storage (five rows per frame), so a fit runs only on its selected rows."""

    def __init__(self, sensors: np.ndarray, tips: np.ndarray) -> None:
        """Flatten the (N,5,4,4) sensor poses and (N,5,3) wrist-frame tip positions."""
        self.frames = len(sensors)
        self.a = sensors[:, :, :3, 3].reshape(-1, 3)
        self.rotation = sensors[:, :, :3, :3].reshape(-1, 3, 3)
        self.b = tips.reshape(-1, 3)
        self.finger = np.tile(np.arange(5), self.frames)

    def residual_rows(self, p: np.ndarray, rows: np.ndarray | slice) -> np.ndarray:
        """Model error in mm, (rows, 3), for ``p`` = rotation vector of Y, translation of Y, five offsets."""
        offsets = p[6:].reshape(5, 3)[self.finger[rows]]
        tip_in_coil = self.a[rows] + np.einsum("nij,nj->ni", self.rotation[rows], offsets)
        return (tip_in_coil - self.b[rows] @ Rotation.from_rotvec(p[:3]).as_matrix().T - p[3:6]) * 1000

    def residual_points(self, p: np.ndarray) -> np.ndarray:
        """Per-fingertip error magnitude in mm, (N, 5)."""
        return np.linalg.norm(self.residual_rows(p, slice(None)), axis=1).reshape(self.frames, 5)

    def residual_frames(self, p: np.ndarray) -> np.ndarray:
        """Per-frame mean of the five fingertip errors, in mm."""
        return self.residual_points(p).mean(axis=1)

    def fit(self, mask: np.ndarray, p0: np.ndarray, max_nfev: int = 200) -> np.ndarray:
        """Robust least squares on a frame mask (N,) or a point mask (N, 5)."""
        points = np.repeat(mask[:, None], 5, axis=1) if mask.ndim == 1 else mask
        # Thinned in time only: every selected finger of a used frame stays.
        rows = np.flatnonzero((points & (np.arange(self.frames) % FIT_STRIDE == 0)[:, None]).ravel())
        return least_squares(
            lambda p: self.residual_rows(p, rows).ravel(),
            p0,
            loss="huber",
            f_scale=HUBER_MM,
            x_scale="jac",
            max_nfev=max_nfev,
        ).x

    @staticmethod
    def start(rotation: np.ndarray) -> np.ndarray:
        """Starting parameters at ``rotation`` with zero translation and offsets."""
        return np.concatenate([Rotation.from_matrix(rotation).as_rotvec(), np.zeros(18)])

    @staticmethod
    def coil_from_wrist(p: np.ndarray) -> np.ndarray:
        """The 4x4 transform Y of a parameter vector."""
        return common.transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6])


def main(argv: list[str] | None = None) -> int:
    """Record (or read) one take, solve it and save a passed result; returns 0 only when every check passed."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog=PROCEDURE, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    record.add_arguments(parser)
    args = parser.parse_args(argv)
    episode = record.episode_for(args, "glove calibration: coil to wrist", False, PROCEDURE)
    times, usable, sensors, tips = load(episode, args.side)
    problem = Problem(sensors, tips)
    print(f"Solving the {args.side} hand: {times[-1]:.0f} s recorded, {usable.mean():.0%} usable...")

    # 1. Initial solution from random starts on a thinned frame set; there is no external initial guess.
    thin = usable & (np.arange(len(times)) % HELD_OUT_STRIDE == 0)
    candidates = [problem.fit(thin, Problem.start(r)) for r in common.random_rotations(RANDOM_STARTS, seed=0)]
    p0 = min(candidates, key=lambda p: common.rms(problem.residual_frames(p)[thin]))

    # 2. Leave-one-segment-out rejection, then point trimming inside the kept segments.
    keep, window, p = common.segment_reject(
        times,
        usable,
        problem.fit,
        problem.residual_frames,
        p0,
        floor_mm=SEG_FLOOR_MM,
        held_out_stride=HELD_OUT_STRIDE,
    )
    kept_frames = usable & keep[window]
    mask, p = common.point_trim(
        np.repeat(kept_frames[:, None], 5, axis=1), problem.fit, problem.residual_points, p
    )

    # 3. Multi-start on the final selection: a second solution that fits as well means the take is ambiguous.
    fits = [p, *(problem.fit(mask, Problem.start(r)) for r in common.random_rotations(RANDOM_STARTS, seed=1))]
    results = [(common.rms(problem.residual_points(q)[mask]), Problem.coil_from_wrist(q)) for q in fits]
    best = min(range(len(fits)), key=lambda i: results[i][0])
    (fit_rms, coil_from_wrist), p = results[best], fits[best]
    offsets = p[6:].reshape(5, 3)

    # 4. Stability (the two time halves must agree) and diversity of the motion that identifies the unknowns.
    half = (times < np.median(times[kept_frames]))[:, None]
    split_deg = common.rotation_between_deg(
        Problem.coil_from_wrist(problem.fit(mask & half, p)), Problem.coil_from_wrist(problem.fit(mask & ~half, p))
    )
    rotations = sensors[kept_frames][:, :, :3, :3]
    span = [common.orientation_span_deg(Rotation.from_matrix(rotations[:, k])) for k in range(5)]
    spread = [np.sqrt(np.linalg.eigvalsh(np.cov(tips[kept_frames, k].T)).max()) * 1000 for k in range(5)]
    still, narrow = common.finger_name(np.argmin(span)), common.finger_name(np.argmin(spread))

    values = dict(
        rms=fit_rms,
        half_split=split_deg,
        dropped=1 - keep.mean(),
        sensor_span=min(span),
        tip_spread=min(spread),
        tip_offset=np.linalg.norm(offsets, axis=1).max() * 1000,
        ambiguity=common.multistart_ambiguity_deg(results, ACCEPT["ambiguity"][2]),
    )
    return common.finish(
        args.results, args.side, "T_coil_wrist", coil_from_wrist, ACCEPT, values, still=still, narrow=narrow
    )


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Find where a glove's main coil sits on its controller.

One fingertip sensor is clamped at a fixed point while the controller and coil are turned around it. Model
(position only; the clamped housing may rotate): ``R_A(t-d) (R_X b(t) + t_X) + t_A(t-d) = y``. ``A`` is the
controller aim pose (``pico_controller/{side}_pose``), ``b`` the clamped sensor's position in the main-coil
frame (``manus_glove/{side}_sensor_pose``), ``X`` the unknown aim-from-coil transform, ``y`` the fixed point and
``d`` the delay of the glove data behind the controller, which is scanned.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

# tools/calibration/ is a plain scripts directory, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as common  # noqa: E402
import record  # noqa: E402

#: What the operator does to record the take; shown by --help and before recording.
PROCEDURE = """What you need:
  glove        not worn, turned on, with the controller mounted on it
  controller   turned on, where the headset can see it
  headset      turned on and tracking
  The other glove and controller are not needed.

Steps:
  1. Clamp one fingertip sensor so it cannot move, and not against metal. Name its finger with --finger.
  2. When recording starts, slowly turn the controller in many directions for about two minutes.
     Keep it moving and do not pull on the clamped sensor.
"""

#: A controller sample is a glitch after a stream gap or a tracking jump; ``pad`` neighbours go with it.
GLITCH = dict(gap_s=0.04, jump_deg=6.0, jump_m=0.012, pad=2)
DELAY_COARSE_MS = np.arange(-100, 301, 10.0)
DELAY_FINE_HALF_MS = 9
SEG_FLOOR_MM = 6.0
HELD_OUT_STRIDE = 4
TRIM_FLOOR_MM = 4.0
HUBER_MM = 3.0
RANDOM_STARTS = 8
_DELAY_ADVICE = "The glove and controller data do not line up. Keep the controller moving the whole time."
#: Each check as (label, comparison, threshold, unit, what to change when it fails). The half-split gate catches
#: clamp slip; noise alone reaches 3-4 degrees on a 30-40 s take.
ACCEPT = {
    "rms": ("fit error", "<", 5.0, "mm", "The clamped sensor moved. Clamp it firmly."),
    "half_split": ("first half vs second half", "<", 5.0, "deg", "The clamp slipped. Clamp it firmly."),
    "dropped": ("part of the take thrown out", "<", 0.30, "", "Check the clamp and turn the controller slower."),
    "glitch": ("controller tracking lost", "<", 0.30, "", "Keep the controller where the headset can see it."),
    "rotation": (
        "how far the controller turned",
        ">",
        90.0,
        "deg",
        "Turn the controller further, in more directions.",
    ),
    "delay_low": ("glove lag, at least", ">", 0.0, "ms", _DELAY_ADVICE),
    "delay_high": ("glove lag, at most", "<", 200.0, "ms", _DELAY_ADVICE),
    "ambiguity": ("other answers that fit as well", "<", 1.0, "deg", "Turn the controller in more directions."),
}


class Track:
    """The controller aim pose over time, interpolated, with its glitch samples marked."""

    def __init__(self, times: np.ndarray, poses: np.ndarray) -> None:
        """Index (N,4,4) aim poses by time and mark stream gaps and tracking jumps."""
        self.times = times
        self.position = poses[:, :3, 3]
        rotation = Rotation.from_matrix(poses[:, :3, :3])
        self.slerp = Slerp(times, rotation)
        # A step is bad when it spans a gap or moves further than a tracked controller can in one sample.
        turned = np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude())
        moved = np.linalg.norm(np.diff(self.position, axis=0), axis=1)
        bad = (np.diff(times) > GLITCH["gap_s"]) | (turned > GLITCH["jump_deg"]) | (moved > GLITCH["jump_m"])
        self.bad = bad.copy()
        for k in range(1, GLITCH["pad"] + 1):
            self.bad[k:] |= bad[:-k]
            self.bad[:-k] |= bad[k:]

    def at(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotation matrices, positions and a usable mask at the ``query`` times."""
        inside = (query > self.times[0]) & (query < self.times[-1])
        clipped = np.clip(query, self.times[0], self.times[-1])
        step = np.clip(np.searchsorted(self.times, clipped), 1, len(self.times) - 1) - 1
        position = np.stack([np.interp(clipped, self.times, self.position[:, k]) for k in range(3)], axis=1)
        return self.slerp(clipped).as_matrix(), position, inside & ~self.bad[step]


def load(episode: Path, side: str, finger: str) -> tuple[Track, np.ndarray, np.ndarray]:
    """The controller aim track, and the clamped sensor's sample times and (N,3) main-coil positions."""
    aim = f"{side}_pose"
    controller = common.read_source(episode, "pico_controller", (aim,), common.CONTROLLER_ROWS_HINT)
    sensor = f"{side}_sensor_pose"
    glove = common.read_source(episode, "manus_glove", (sensor,), common.GLOVE_ROWS_HINT)
    # The recorder zeroes an unreported sensor set, so a real transform ends in 1.
    reported = (glove[sensor][:, :, 3, 3] == 1).all(axis=1)
    times = glove["capture_time"][reported]
    common.require_take(times, f"{side} fingertip-sensor data", common.SENSOR_HINT)
    track = Track(controller["capture_time"], controller[aim])
    return track, times, glove[sensor][reported, common.FINGERS.index(finger), :3, 3]


def unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation and translation of X, and the fixed point y, of a parameter vector."""
    return Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6], p[6:9]


def aim_from_coil(p: np.ndarray) -> np.ndarray:
    """The 4x4 transform X of a parameter vector."""
    rotation, translation, _ = unpack(p)
    return common.transform(rotation, translation)


class Problem:
    """The fit at one delay: controller poses already interpolated onto the sensor times."""

    def __init__(self, aim_rotation: np.ndarray, aim_position: np.ndarray, b: np.ndarray) -> None:
        """Store the per-sample aim pose and sensor position."""
        self.aim_rotation, self.aim_position, self.b = aim_rotation, aim_position, b

    def error(self, p: np.ndarray, rows: np.ndarray | slice = slice(None)) -> np.ndarray:
        """Offset in mm, (rows, 3), of the sensor carried into the room frame by ``p`` from the fixed point."""
        rotation, translation, point = unpack(p)
        in_aim = self.b[rows] @ rotation.T + translation
        return (np.einsum("nij,nj->ni", self.aim_rotation[rows], in_aim) + self.aim_position[rows] - point) * 1000

    def resid(self, p: np.ndarray) -> np.ndarray:
        """Per-sample distance from the fixed point, in mm."""
        return np.linalg.norm(self.error(p), axis=1)

    def fit(self, mask: np.ndarray, p0: np.ndarray, max_nfev: int = 400) -> np.ndarray:
        """Robust least squares on the masked samples."""
        rows = np.flatnonzero(mask)
        return least_squares(
            lambda p: self.error(p, rows).ravel(),
            p0,
            loss="huber",
            f_scale=HUBER_MM,
            x_scale="jac",
            max_nfev=max_nfev,
        ).x

    def start(self, rotation: np.ndarray) -> np.ndarray:
        """Starting parameters at ``rotation``, zero translation and the median point that implies."""
        p = np.concatenate([Rotation.from_matrix(rotation).as_rotvec(), np.zeros(6)])
        p[6:9] = np.median(self.error(p), axis=0) / 1000
        return p


def quick_rms(problem: Problem, ok: np.ndarray, p0: np.ndarray) -> tuple[float, np.ndarray]:
    """One robust fit and one trim: cheap enough to rank starts and delays, not a final solution."""
    p = problem.fit(ok, p0)
    residual = problem.resid(p)
    mask = ok & (residual < max(common.TRIM_K * np.median(residual[ok]), TRIM_FLOOR_MM))
    p = problem.fit(mask, p)
    return common.rms(problem.resid(p)[mask]), p


def scan_delay(
    at_delay: Callable[[float], tuple[Problem, np.ndarray]], p0: np.ndarray
) -> tuple[float, np.ndarray]:
    """The delay in ms that minimises the quick rms, and the solution there.

    A coarse grid, a 1 ms grid around its best, then a parabola through the fine minimum.
    ``at_delay(ms)`` returns the (problem, usable mask) at that delay.
    """
    scan = {}
    for delay in DELAY_COARSE_MS:
        scan[float(delay)], p0 = quick_rms(*at_delay(delay), p0)
    coarse = min(scan, key=scan.get)
    for delay in np.arange(coarse - DELAY_FINE_HALF_MS, coarse + DELAY_FINE_HALF_MS + 0.1, 1.0):
        scan[float(delay)] = quick_rms(*at_delay(delay), p0)[0]
    best = min(scan, key=scan.get)
    near = [delay for delay in scan if abs(delay - best) <= 3]
    curve = np.polyfit(near, [scan[delay] for delay in near], 2)
    # A parabola without a minimum keeps the grid's best.
    if curve[0] > 0:
        best = float(np.clip(-curve[1] / (2 * curve[0]), best - 3, best + 3))
    return best, p0


def main(argv: list[str] | None = None) -> int:
    """Record (or read) one take, solve it and save a passed result; returns 0 only when every check passed."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog=PROCEDURE, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    record.add_arguments(parser)
    parser.add_argument(
        "--finger", choices=common.FINGERS, required=True, help="the finger whose sensor is clamped"
    )
    args = parser.parse_args(argv)
    episode = record.episode_for(args, "glove calibration: aim to coil", True, PROCEDURE)
    track, sensor_times, b = load(episode, args.side, args.finger)
    times = sensor_times - sensor_times[0]

    def at_delay(delay_ms: float) -> tuple[Problem, np.ndarray]:
        """The problem and usable mask with the controller read ``delay_ms`` before each sensor sample."""
        aim_rotation, aim_position, ok = track.at(sensor_times - delay_ms / 1000)
        return Problem(aim_rotation, aim_position, b), ok

    problem, ok = at_delay(0.0)
    common.require_take(sensor_times[ok], f"{args.side} controller tracking", common.CONTROLLER_ROWS_HINT)
    print(f"Solving the {args.side} hand: {times[-1]:.0f} s recorded, {ok.mean():.0%} usable...")

    # 1. Initial solution at zero delay from random starts; there is no external initial guess.
    starts = [problem.start(r) for r in common.random_rotations(RANDOM_STARTS, seed=0)]
    p0 = min((quick_rms(problem, ok, start) for start in starts), key=lambda candidate: candidate[0])[1]

    # 2. The glove data lags the controller by an unknown delay.
    delay_ms, p0 = scan_delay(at_delay, p0)
    problem, ok = at_delay(delay_ms)

    # 3. Leave-one-segment-out rejection, then point trimming inside the kept segments.
    keep, window, p = common.segment_reject(
        times, ok, problem.fit, problem.resid, p0, floor_mm=SEG_FLOOR_MM, held_out_stride=HELD_OUT_STRIDE
    )
    mask, p = common.point_trim(ok & keep[window], problem.fit, problem.resid, p, TRIM_FLOOR_MM)

    # 4. Multi-start on the final selection: a second solution that fits as well means the take is ambiguous.
    fits = [p, *(problem.fit(mask, problem.start(r)) for r in common.random_rotations(RANDOM_STARTS, seed=1))]
    results = [(common.rms(problem.resid(q)[mask]), aim_from_coil(q)) for q in fits]
    best = min(range(len(fits)), key=lambda i: results[i][0])
    (fit_rms, x), p = results[best], fits[best]

    # 5. Stability (the two time halves must agree) and how far the controller turned around the clamp.
    half = times < np.median(times[mask])
    split_deg = common.rotation_between_deg(
        aim_from_coil(problem.fit(mask & half, p)), aim_from_coil(problem.fit(mask & ~half, p))
    )
    span_deg = common.orientation_span_deg(Rotation.from_matrix(problem.aim_rotation[mask]))

    values = dict(
        rms=fit_rms,
        half_split=split_deg,
        dropped=1 - keep.mean(),
        glitch=1 - ok.mean(),
        rotation=span_deg,
        delay_low=delay_ms,
        delay_high=delay_ms,
        ambiguity=common.multistart_ambiguity_deg(results, ACCEPT["ambiguity"][2]),
    )
    return common.finish(args.results, args.side, "T_controller_manus", x, ACCEPT, values)


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rules shared by the Manus calibration solvers, so that no take needs a per-run judgement.

Windows are rejected by a leave-one-segment-out score (a window is scored by a fit that did not see it, so a
bad window cannot pull the model toward itself), points are trimmed inside the kept windows, and a calibration
passes only if every row of its acceptance table does.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

#: The recorded fingertip-sensor order of ``manus_glove/{side}_sensor_pose``.
FINGERS = ("thumb", "index", "middle", "ring", "little")
#: A take with less usable data than this has too few windows for the selection rules.
MIN_TAKE_S = 20.0
SEG_WINDOW_S = 5.0
#: A window is dropped when its held-out rms exceeds this many times the median of the kept windows.
SEG_FACTOR = 1.5
SEG_MAX_PASSES = 6
#: Evaluation cap of a held-out fit, which is warm-started from the current solution.
SEG_HELD_OUT_NFEV = 30
TRIM_K = 2.5
TRIM_PASSES = 3

#: The station calibration file passed results go into; a recording copies it into every session.
RESULTS_NAME = "wrist_calibration_offsets.json"

#: Why a full recording can lack a source, as told to the operator.
GLOVE_ROWS_HINT = "A full recording has glove data only while both gloves are on and connected."
CONTROLLER_ROWS_HINT = "A full recording has controller data only while the headset sees both controllers."
SENSOR_HINT = "Check that the glove is on and connected."


def rms(values: np.ndarray) -> float:
    """Root mean square of ``values``."""
    return float(np.sqrt(np.mean(np.square(values))))


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """The 4x4 homogeneous transform of a rotation matrix and a translation."""
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def rotation_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between the rotations of two 4x4 transforms."""
    return float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))


def orientation_span_deg(rotations: Rotation) -> float:
    """Twice the 95th-percentile angle of ``rotations`` from their mean: how far a body turned."""
    return 2 * float(np.percentile(np.degrees((rotations.mean().inv() * rotations).magnitude()), 95))


def random_rotations(count: int, seed: int) -> np.ndarray:
    """``count`` reproducible random rotation matrices, the solvers' only starting points."""
    return Rotation.random(count, random_state=seed).as_matrix()


def segment_reject(
    times: np.ndarray,
    usable: np.ndarray,
    fit: Callable[..., np.ndarray],
    resid: Callable[[np.ndarray], np.ndarray],
    p0: np.ndarray,
    *,
    floor_mm: float,
    held_out_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-segment-out rejection of ``SEG_WINDOW_S`` windows.

    ``fit(frame_mask, p0[, max_nfev])`` returns parameters and ``resid(p)`` a per-frame residual in mm. A
    window is dropped when its held-out rms exceeds ``max(SEG_FACTOR * median, floor_mm)`` over the kept
    windows. Returns the per-window keep mask, each frame's window index and the parameters.
    """
    window = (times // SEG_WINDOW_S).astype(int)
    count = window.max() + 1
    # A window with too few usable frames cannot be scored, so it is never kept.
    keep = np.array([(usable & (window == w)).sum() >= 20 for w in range(count)])
    # Held-out fits are thinned for speed; the scores use every frame.
    thinned = np.arange(len(times)) % held_out_stride == 0
    p = fit(usable & keep[window], p0)
    for _ in range(SEG_MAX_PASSES):
        scores = np.full(count, np.nan)
        for w in np.flatnonzero(keep):
            held_out = fit(usable & keep[window] & (window != w) & thinned, p, SEG_HELD_OUT_NFEV)
            scores[w] = rms(resid(held_out)[usable & (window == w)])
        # The floor stops the relative threshold from ratcheting down once the data is clean.
        kept = keep & ~(scores > max(SEG_FACTOR * np.nanmedian(scores), floor_mm))
        if np.array_equal(kept, keep):
            break
        keep = kept
        p = fit(usable & keep[window], p)
    return keep, window, p


def point_trim(
    mask: np.ndarray,
    fit: Callable[..., np.ndarray],
    resid: Callable[[np.ndarray], np.ndarray],
    p: np.ndarray,
    floor_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim residuals above ``max(TRIM_K * rms, floor_mm)`` inside ``mask``; returns the mask and parameters."""
    base = mask
    for _ in range(TRIM_PASSES):
        p = fit(mask, p)
        residual = resid(p)
        # Trimming restarts from the full selection, so a point can come back once the fit improves.
        trimmed = base & (residual < max(TRIM_K * rms(residual[mask]), floor_mm))
        if np.array_equal(trimmed, mask):
            break
        mask = trimmed
    return mask, p


def multistart_ambiguity_deg(results: list[tuple[float, np.ndarray]], tolerance_deg: float) -> float:
    """Largest rotation between the best (rms, transform) and another that fits within 5% of it.

    Differences up to ``tolerance_deg`` are the same solution and count as 0.
    """
    best_rms, best = min(results, key=lambda result: result[0])
    angles = [rotation_between_deg(best, other) for other_rms, other in results if other_rms <= 1.05 * best_rms]
    return max((angle for angle in angles if angle > tolerance_deg), default=0.0)


def resolve_episode(path: Path) -> Path:
    """``path`` itself when it is an episode directory, or the last episode of the session directory it names."""
    indices = [int(child.name[8:]) for child in path.glob("episode_*") if child.name[8:].isdigit()]
    if not indices:
        return path
    episode = path / f"episode_{max(indices)}"
    print(f"Using the session's last episode: {episode}")
    return episode


def read_source(episode: Path, source: str, fields: tuple[str, ...], hint: str) -> dict[str, np.ndarray]:
    """``capture_time`` and the named fields of one numeric source of a recorded (not time-aligned) episode.

    ``hint`` tells the operator what makes the recorder write this source; it ends the refusal of an episode
    that has no rows of it.
    """
    if not (episode / source).is_dir():
        raise SystemExit(f"{episode} has no {source} data. {hint}")
    group = zarr.open_group(str(episode / source), mode="r")
    stored = {name for name, _ in group.arrays()}
    # A missing field is never substituted: the episode was recorded by a version that did not store it.
    missing = [name for name in fields if name not in stored]
    if missing:
        raise SystemExit(f"{episode}: {source} has no {', '.join(missing)}. Record a new take.")
    data = {name: group[name][:] for name in ("capture_time", *fields)}
    if len(data["capture_time"]) == 0:
        raise SystemExit(f"{episode} has no {source} data. {hint}")
    return data


def require_take(usable_times: np.ndarray, what: str, hint: str) -> None:
    """Refuse a take whose usable samples span less than ``MIN_TAKE_S``."""
    seconds = float(np.ptp(usable_times)) if len(usable_times) else 0.0
    if seconds < MIN_TAKE_S:
        raise SystemExit(f"Only {seconds:.0f} s of {what}; a take needs at least {MIN_TAKE_S:.0f} s. {hint}")


def finger_name(index: int) -> str:
    """A finger as an operator would say it."""
    return FINGERS[index] if index == 0 else f"{FINGERS[index]} finger"


def finish(
    results: Path,
    side: str,
    name: str,
    result: np.ndarray,
    accept: dict[str, tuple],
    values: dict[str, float],
    **names: str,
) -> int:
    """Judge ``values`` against an acceptance table, print it, save a passed result and return the exit status.

    ``accept`` maps a check to (label, ``<`` or ``>``, threshold, unit, advice); a failed row prints its advice,
    which says what to change in the next take. ``names`` fill the ``{...}`` fields of labels and advice. A
    passed 4x4 ``result`` replaces ``results[side][name]`` and leaves the rest of the file as it was; a failed
    one never replaces an earlier result.
    """
    passed = True
    for key, (label, op, threshold, unit, advice) in accept.items():
        value = float(values[key])
        ok = value < threshold if op == "<" else value > threshold
        passed = passed and ok
        label = label.format(**names)
        print(f"  [{'ok' if ok else 'FAIL':4s}] {label:40s} {value:8.2f} {unit:3s} (need {op} {threshold:g})")
        if not ok:
            print(f"         -> {advice.format(**names)}")
    if not passed:
        print("  => FAIL. Nothing was saved; record again.")
        return 1
    saved = json.loads(results.read_text()) if results.is_file() else {}
    entry = saved.setdefault(side, {}) if isinstance(saved, dict) else None
    if not isinstance(entry, dict):
        raise SystemExit(f"{results} is not laid out by side. Pass --results with another file.")
    entry[name] = result.tolist()
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    print(f"  => PASS. Saved as {side} {name} in {results}")
    return 0

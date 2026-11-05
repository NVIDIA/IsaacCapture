# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Quality control for a hand-reconstruction clip: keep it, or drop it.

Decides from the delivered 3D hands, the intrinsics and the frame size. Never
opens the video, so it costs milliseconds and can sweep a dataset cheaply.

Scores the SOMA-25 joints the model predicts, in the layout it predicts them
(topology from ``SOMA_constants``). Nothing is remapped to MANO-21 on the way
in: that would drop the four palm joints for no gain. Numpy only -- no torch,
no OpenCV, no video decoder.

Usage
-----
    from pathlib import Path

    from inference.postprocessing.clip_quality_control import check_clip, load_intrinsics

    K, image_size = load_intrinsics("undistorted_intrinsics.json")
    verdict = check_clip("left_results.npz", "right_results.npz",
                         K=K, image_size=image_size, max_bad_frac=0.02)

    verdict.drop            # True -> throw this clip away. That is the answer.
    verdict.to_json()       # why, as JSON (~0.7 KB) -- store it for dropped clips
    verdict.summary()       # the same, as text, for a log line
    verdict.hands[s].flags  # per-frame flags, to drop frames instead of a clip

    if verdict.drop:        # keep the evidence next to the clip you removed
        Path(f"qc/{verdict.clip}.json").write_text(verdict.to_json(indent=2))

``max_bad_frac`` is the share of frame-hands that may be flagged before the clip
is marked for dropping. The default is 0: any flagged prediction fails QC.
These checks are heuristics, so inspect the reports and visualizations before
choosing a tolerance for a new dataset. The checker does not delete outputs.

``to_json()`` carries counts per rule and per hand, not frame ids; pass
``with_frames=True`` to add them (~18 KB on a bad clip).

For joints already in memory, call ``check_joints(left, right, ...)``.

Checks
------
Per hand:   impossible_motion   >6 m/s, or wrist above 150 m/s^2 (15 g)
            root_off_frame      the wrist projects outside the image
            size_issue          hand size stepped >5%, or sits >15% off the
                                clip's median. One clip is one person.
            handedness          chirality disagrees with the file: a mirrored
                                or mislabelled hand
Both hands: hands_collide       wrists within a quarter hand-span
            lr_size_mismatch    the two hands differ in size by >15%

Delivery checks fail regardless of ``max_bad_frac``: there must be finite
hand predictions, the hands must cover the same frames, and neither may be
absent from >10% of them when frame coverage is known.

Notes
-----
Thresholds come from physics, from a published QA spec, or from a measurement
recorded at the constant -- never from a percentile.
Motion and size checks use distances and rates. The handedness check uses
signed chirality and expects the pipeline's camera-coordinate convention.

``root_off_frame`` needs the real camera. The pipeline's ``estimate_K`` fallback
is double the true focal on typical 1920x1080 egocentric footage and flags two
thirds of frames, so ``check_clip`` refuses a K that matches it.

Blind to a hand that is smooth, correctly sized and simply in the wrong place --
that needs the frames.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from inference.postprocessing.soma_constants import (
    SOMA25_INDEX_MCP,
    SOMA25_LITTLE_MCP,
    SOMA25_N_JOINTS,
    SOMA25_PARENT,
    SOMA25_THUMB_MCP,
    SOMA25_WRIST,
)

PathLike = str | Path


class CameraProjection(Protocol):
    """Camera model projecting camera-frame 3D points into source-image pixels."""

    def project(self, points: np.ndarray) -> np.ndarray: ...


CameraIntrinsics = Sequence[Sequence[float]] | np.ndarray | CameraProjection

N_JOINTS = SOMA25_N_JOINTS

#: Every joint but the wrist, and its parent: the 24 bones. Scored in the layout
#: the model predicts, with no remap to MANO-21 -- that would throw away the four
#: palm joints, which is exactly where a broken shape head shows first.
BONE_CHILD = np.arange(1, N_JOINTS)
BONE_PARENT = np.asarray(SOMA25_PARENT)[BONE_CHILD]

# ---------------------------------------------------------------------------
# Thresholds. Each is anchored to physics or to a published limit, never to a
# percentile of whatever corpus happens to be at hand.
# ---------------------------------------------------------------------------

#: Speed no hand exceeds, m/s. A 20 cm/frame limit at 30 fps, held per
#: second so frame rate cannot change what it means.
MAX_SPEED_MPS = 6.0

#: Acceleration no arm produces, m/s^2. Hand motion peaks at 10-15 g; on
#: real recordings, healthy checkpoints hit 245 at worst, a broken crop
#: box 1367.
MAX_ACCEL_MPS2 = 150.0

#: Per-frame change in total bone length, %. Measurement floor is ~0.3%, p99 is
#: 3.7%, so 5% is clear of noise.
MAX_SIZE_STEP_PCT = 5.0

#: Deviation from the clip's median hand size. Adults span about +-12% in total
#: bone length, so a wider swing inside one clip is the estimator, not the
#: person. Relative, so it does not care that the total bone length differs
#: across skeleton topologies (e.g. SOMA-25 vs. a 21-joint convention).
MAX_SIZE_DEVIATION = 0.15

#: Chirality: cross(index_mcp, pinky_mcp) . thumb_mcp about the wrist is
#: negative for a left hand, positive for a right. 69 clips, zero exceptions.
_CHIRALITY_SIGN = {"left": -1.0, "right": +1.0}

#: Two wrists closer than this are not two hands. In hand-spans, not cm, so it
#: holds at any hand size; clasped hands sit about one span apart.
MIN_WRIST_SEPARATION_SPANS = 0.25

#: One person's two hands match in size. Healthy clip measures 4.5%.
MAX_LR_SIZE_MISMATCH = 0.15

#: Absent for more than this share of the clip: incomplete, not occluded.
MAX_MISSING_FRAC = 0.10

RULES = ("impossible_motion", "root_off_frame", "size_issue", "handedness")
PAIR_RULES = ("hands_collide", "lr_size_mismatch")


@dataclass
class HandQuality:
    """Per-frame flags for one hand, plus the counts."""

    n_frames: int = 0
    flags: dict[str, np.ndarray] = field(default_factory=dict)
    #: Frame id of each row, so a flag can name the frame rather than the row.
    #: Rows are not frame numbers whenever a hand skips frames.
    frame_ids: np.ndarray | None = None

    @property
    def any_flag(self) -> np.ndarray:
        if not self.flags:
            return np.zeros(self.n_frames, bool)
        out = np.zeros(self.n_frames, bool)
        for v in self.flags.values():
            out |= v
        return out

    def counts(self) -> dict[str, int]:
        return {k: int(v.sum()) for k, v in self.flags.items()}

    def flagged_frames(self) -> dict[str, list[int]]:
        """``{rule: [frame id, ...]}`` for the rules that fired."""
        out = {}
        for rule, mask in self.flags.items():
            rows = np.flatnonzero(mask)
            if not len(rows):
                continue
            out[rule] = (
                [int(i) for i in rows]
                if self.frame_ids is None
                else [int(self.frame_ids[i]) for i in rows]
            )
        return out


@dataclass
class ClipQuality:
    """The answer: drop or keep, and everything behind it."""

    drop: bool
    reasons: list[str]
    bad_frac: float
    n_frames: int
    #: Clip name, for logs that hold more than one clip.
    clip: str = ""
    hands: dict[str, HandQuality] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)
    missing_frames: dict[str, list[int]] = field(default_factory=dict)

    def flagged_frames(self) -> dict[str, dict[str, list[int]]]:
        """``{side: {rule: [frame id, ...]}}`` -- every flag, by name and frame."""
        return {side: h.flagged_frames() for side, h in self.hands.items()}

    def to_dict(self, with_frames: bool = False) -> dict:
        """The verdict as JSON-safe data.

        Args:
            with_frames: include flagged and missing frame IDs. Off by default: counts
                are ~0.7 kB per clip and the ids take it to ~18 kB on a bad one,
                not worth carrying in an index most rows of which are KEEP.

        Returns:
            Plain dict -- no numpy types -- ready for ``json.dump``.
        """
        out = {
            "clip": self.clip,
            "drop": bool(self.drop),
            "bad_frac": round(float(self.bad_frac), 6),
            "n_frames": int(self.n_frames),
            "reasons": list(self.reasons),
            "counts": {k: int(v) for k, v in sorted(self.counts.items()) if v},
            "hands": {
                side: {
                    "n_frames": int(h.n_frames),
                    "counts": {k: int(v) for k, v in sorted(h.counts().items()) if v},
                }
                for side, h in sorted(self.hands.items())
            },
            "missing": {k: int(v) for k, v in sorted(self.missing.items())},
        }
        if with_frames:
            out["frames"] = self.flagged_frames()
            out["missing_frames"] = self.missing_frames
        return out

    def to_json(self, with_frames: bool = False, **kwargs) -> str:
        """``to_dict`` as a JSON string; ``kwargs`` go to ``json.dumps``."""
        return json.dumps(self.to_dict(with_frames), **kwargs)

    def summary(self) -> str:
        """Human-readable verdict: what, why, and how many frames per issue.

        Counts only, ~0.6 kB. The frame ids stay in ``flagged_frames()`` for a
        caller that wants to open the pictures -- writing them here would put
        ~18 kB in every log line.
        """
        name = f"{self.clip}  " if self.clip else ""
        out = [
            f"{'DROP' if self.drop else 'KEEP'}  {name}"
            f"{self.bad_frac * 100:.2f}% of {self.n_frames} frame-hands flagged"
        ]
        out += [f"  {r}" for r in self.reasons]
        for side, h in sorted(self.hands.items()):
            for rule, n in sorted(h.counts().items()):
                if n:
                    out.append(f"  {side:5s} {rule:18s} {n:5d} frames")
        return "\n".join(out)


def load_intrinsics(path: PathLike) -> tuple[np.ndarray, tuple[int, int]]:
    """``(K, (width, height))`` from an undistorted-intrinsics JSON.

    Reads the calibration rather than reconstructing it, because the two ways of
    guessing are both wrong in ways that matter here:

    * ``estimate_K``'s ``focal = max(W, H)`` is the pipeline's fallback when no
      intrinsics are supplied. Whenever the true focal is meaningfully smaller
      than that heuristic, it places every hand farther than its real
      distance and projects it farther from the principal point -- enough,
      on its own, to make ``root_off_frame`` fire on a large share of
      frame-hands: a wrong camera reported as a broken delivery.
    * Assuming the principal point is the image centre. A calibration whose
      true ``cx``/``cy`` is off-center by even a modest offset shifts every
      projected wrist toward one edge.

    Accepts either the ``camera_matrix.matrix_3x3`` block or bare
    ``fx``/``fy``/``cx``/``cy``.
    """
    import json

    with open(Path(path)) as f:
        d = json.load(f)
    cam = d.get("camera_matrix", d)
    if "matrix_3x3" in cam:
        K = np.asarray(cam["matrix_3x3"], dtype=np.float64)
    else:
        try:
            K = np.array(
                [
                    [float(cam["fx"]), 0.0, float(cam["cx"])],
                    [0.0, float(cam["fy"]), float(cam["cy"])],
                    [0.0, 0.0, 1.0],
                ]
            )
        except KeyError as e:
            raise KeyError(f"{path}: no matrix_3x3 and no {e.args[0]}") from None
    if K.shape != (3, 3):
        raise ValueError(f"{path}: intrinsics must be 3x3, got {K.shape}")
    w, h = d.get("image_width"), d.get("image_height")
    if w is None or h is None:
        raise KeyError(f"{path}: expected image_width and image_height")
    return K, (int(w), int(h))


def looks_like_estimated_K(K, image_size) -> bool:
    """True when ``K`` is the no-intrinsics fallback rather than a calibration.

    ``estimate_K`` builds ``focal = max(W, H)`` with the principal point at the
    image centre. That is a placeholder, not a camera, and every metric derived
    from it is wrong by the ratio to the true focal -- so it is worth naming
    rather than silently screening against.
    """
    if K is None or image_size is None or callable(getattr(K, "project", None)):
        return False
    K = np.asarray(K, float)
    w, h = float(image_size[0]), float(image_size[1])
    f = max(w, h)
    return bool(
        abs(K[0, 0] - f) < 1e-6
        and abs(K[1, 1] - f) < 1e-6
        and abs(K[0, 2] - w / 2) < 1e-6
        and abs(K[1, 2] - h / 2) < 1e-6
    )


def _load(path: PathLike) -> tuple[np.ndarray, np.ndarray | None]:
    """Camera-frame SOMA-25 joints ``(F, 25, 3)`` in metres, and frame ids.

    A hand that never got a box is legitimate, not corrupt: ``num_frames == 0``
    with no ``joints`` array loads as an empty ``(0, 25, 3)``.
    """
    with np.load(Path(path), allow_pickle=False) as d:
        if "joints" not in d:
            if "num_frames" in d and int(np.asarray(d["num_frames"])) == 0:
                return np.zeros((0, N_JOINTS, 3)), np.zeros((0,), np.int64)
            raise KeyError(f"{path}: expected 'joints', got {list(d.keys())}")
        j = np.asarray(d["joints"], dtype=np.float64)
        if j.ndim != 3 or j.shape[1:] != (N_JOINTS, 3):
            raise ValueError(
                f"{path}: expected SOMA-25 joints (F,{N_JOINTS},3), got {j.shape}"
            )
        if "cam_t" in d:  # root-relative -> camera frame
            j = j + np.asarray(d["cam_t"], dtype=np.float64)[:, None, :]
        ids = None
        if "frame_names" in d:
            ids = np.array(
                [
                    int(str(s).strip().lstrip("0") or 0)
                    for s in np.asarray(d["frame_names"]).ravel()
                ],
                np.int64,
            )
    return j, ids


def _steps(
    values: np.ndarray, present: np.ndarray, frame_ids=None
) -> tuple[np.ndarray, np.ndarray]:
    """Displacement between consecutive present frames, and the later index.

    Skips frames either side of a dropout: a gap is not motion.
    """
    ok = present[:-1] & present[1:]
    if frame_ids is not None:
        ok &= np.diff(frame_ids) == 1
    if not ok.any():
        return np.empty(0), np.empty(0, np.int64)
    d = np.linalg.norm(np.diff(values, axis=0), axis=-1)
    if d.ndim == 2:
        d = d.max(axis=1)
    idx = np.flatnonzero(ok)
    return d[idx], idx + 1


def _check_hand(
    joints: np.ndarray,
    K: CameraIntrinsics | None,
    image_size: tuple[int, int] | None,
    fps: float,
    side: str = "",
    frame_ids=None,
) -> HandQuality:
    n = len(joints)
    v = HandQuality(n_frames=n, flags={r: np.zeros(n, bool) for r in RULES})
    if n == 0:
        return v
    present = np.isfinite(joints).all(axis=(1, 2))
    if not present.any():
        return v

    # -- impossible motion ---------------------------------------------------
    wrist = joints[:, 0, :]
    local = joints - wrist[:, None, :]
    dw, iw = _steps(wrist, present, frame_ids)
    dl, il = _steps(local, present, frame_ids)
    limit_m = MAX_SPEED_MPS / fps  # metres per frame
    if len(dw):
        v.flags["impossible_motion"][iw[dw > limit_m]] = True
    if len(dl):
        v.flags["impossible_motion"][il[dl > limit_m]] = True
    ok3 = present[:-2] & present[1:-1] & present[2:]
    if frame_ids is not None:
        consecutive = np.diff(frame_ids) == 1
        ok3 &= consecutive[:-1] & consecutive[1:]
    if ok3.any():
        acc = np.linalg.norm(np.diff(wrist, 2, axis=0), axis=1) * fps**2
        hit = np.flatnonzero(ok3)
        v.flags["impossible_motion"][hit[acc[hit] > MAX_ACCEL_MPS2] + 1] = True

    # -- the wrist projects outside the image ---------------------------------
    if K is not None and image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        z = wrist[:, 2]
        safe = present & np.isfinite(z) & (z > 1e-6)
        if safe.any():
            if callable(getattr(K, "project", None)):
                uv = K.project(wrist[safe])
            else:
                uv = (wrist[safe] / z[safe, None]) @ np.asarray(K, float).T
            outside = ~(
                (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            )
            v.flags["root_off_frame"][np.flatnonzero(safe)[outside]] = True
        # Behind the camera is not in the picture either.
        v.flags["root_off_frame"][present & ~safe] = True

    # -- handedness ----------------------------------------------------------
    # Invisible to the other rules: a mirrored hand moves and sizes plausibly.
    if side in _CHIRALITY_SIGN:
        wrist = joints[:, SOMA25_WRIST]
        n = np.cross(
            joints[:, SOMA25_INDEX_MCP] - wrist, joints[:, SOMA25_LITTLE_MCP] - wrist
        )
        chir = np.einsum("ij,ij->i", n, joints[:, SOMA25_THUMB_MCP] - wrist)
        wrong = present & np.isfinite(chir) & (np.sign(chir) != _CHIRALITY_SIGN[side])
        v.flags["handedness"] = wrong

    # -- hand size -------------------------------------------------------------
    bones = np.linalg.norm(joints[:, BONE_CHILD] - joints[:, BONE_PARENT], axis=2)
    good = present & np.isfinite(bones).all(axis=1)
    if good.sum() >= 10:
        total = bones.sum(axis=1)
        mid = float(np.median(total[good]))
        if mid > 0:
            v.flags["size_issue"] |= good & (
                np.abs(total - mid) / mid > MAX_SIZE_DEVIATION
            )
            idx = np.flatnonzero(good)
            step = (
                np.diff(idx if frame_ids is None else np.asarray(frame_ids)[idx]) == 1
            )
            d = np.abs(np.diff(total[idx])) / mid * 100.0
            v.flags["size_issue"][idx[1:][step & (d > MAX_SIZE_STEP_PCT)]] = True
    return v


def _check_pair(left: np.ndarray, lids, right: np.ndarray, rids) -> dict:
    """Flags needing both hands on one frame.

    Returns ``{rule: {side: row indices}}``, so neither hand owns a joint fault.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    if len(left) == 0 or len(right) == 0:
        return out
    if lids is None or rids is None:
        n = min(len(left), len(right))
        rl, rr = np.arange(n), np.arange(n)
    else:
        common = np.intersect1d(lids, rids)
        if not len(common):
            return out
        li = {int(f): k for k, f in enumerate(lids)}
        ri = {int(f): k for k, f in enumerate(rids)}
        rl = np.array([li[int(f)] for f in common])
        rr = np.array([ri[int(f)] for f in common])

    L, R = left[rl], right[rr]
    ok = np.isfinite(L).all(axis=(1, 2)) & np.isfinite(R).all(axis=(1, 2))
    if not ok.any():
        return out

    # Two wrists cannot occupy the same place.
    span = (
        np.linalg.norm(L - L[:, :1], axis=2).max(axis=1)
        + np.linalg.norm(R - R[:, :1], axis=2).max(axis=1)
    ) / 2.0
    sep = np.linalg.norm(L[:, 0] - R[:, 0], axis=1) / np.maximum(span, 1e-9)
    hit = ok & np.isfinite(sep) & (sep < MIN_WRIST_SEPARATION_SPANS)
    if hit.any():
        out["hands_collide"] = {"left": rl[hit], "right": rr[hit]}

    # One person, two hands: they are the same size.
    sl = np.linalg.norm(L[:, BONE_CHILD] - L[:, BONE_PARENT], axis=2).sum(axis=1)
    sr = np.linalg.norm(R[:, BONE_CHILD] - R[:, BONE_PARENT], axis=2).sum(axis=1)
    mid = (sl + sr) / 2.0
    gap = np.abs(sl - sr) / np.maximum(mid, 1e-9)
    hit2 = ok & np.isfinite(gap) & (gap > MAX_LR_SIZE_MISMATCH)
    if hit2.any():
        out["lr_size_mismatch"] = {"left": rl[hit2], "right": rr[hit2]}
    return out


def check_hand_results(
    results_per_hand: dict,
    *,
    num_frames: int,
    K: CameraIntrinsics,
    image_size: tuple[int, int],
    fps: float = 30.0,
    max_bad_frac: float = 0.0,
    clip: str = "",
) -> ClipQuality:
    """Check a complete video's in-memory ``EgoHandPredictor`` outputs.

    Call after collecting all frame batches. Result keys must be zero-based
    video frame IDs (numeric strings are accepted). ``pred_keypoints_3d`` is
    wrist-relative SOMA-25 geometry in metres; adding ``transl`` puts it in the
    camera frame. ``K`` is a pinhole matrix or a camera with a ``project``
    method (FISHEYE624/KB4). It and ``image_size`` must describe the source
    frames, including any undistortion or pixel-aspect correction; perspective
    crops already have their predictions rotated back into source-camera axes.

    Both hands are evaluated. ``num_frames`` is the number of decoded frames
    actually processed, including frames with no detections; this lets QC
    count missing detections at the beginning and end of the clip as well.
    Predictions are never modified.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if looks_like_estimated_K(K, image_size):
        raise ValueError(
            "QC requires real camera intrinsics, not the estimate_K fallback"
        )
    joints, ids = {}, {}
    for side in ("left", "right"):
        frames = sorted(
            (
                (int(frame_id), result)
                for frame_id, result in results_per_hand.get(side, {}).items()
            ),
            key=lambda item: item[0],
        )
        ids[side] = np.asarray([frame_id for frame_id, _ in frames], dtype=np.int64)
        if (
            len(np.unique(ids[side])) != len(frames)
            or np.any(ids[side] < 0)
            or np.any(ids[side] >= num_frames)
        ):
            raise ValueError(f"{side}: expected unique frame IDs in [0, {num_frames})")
        joints[side] = (
            np.stack(
                [
                    np.asarray(result["pred_keypoints_3d"], dtype=np.float64)
                    + np.asarray(result["transl"], dtype=np.float64)[None, :]
                    for _, result in frames
                ]
            )
            if frames
            else np.empty((0, N_JOINTS, 3), dtype=np.float64)
        )
    return check_joints(
        joints["left"],
        joints["right"],
        left_ids=ids["left"],
        right_ids=ids["right"],
        K=K,
        image_size=image_size,
        fps=fps,
        max_bad_frac=max_bad_frac,
        clip=clip,
        expected_num_frames=num_frames,
    )


def check_clip(
    left_npz: PathLike,
    right_npz: PathLike,
    *,
    K: CameraIntrinsics | None = None,
    image_size: tuple[int, int] | None = None,
    max_bad_frac: float = 0.0,
    fps: float = 30.0,
    clip: str = "",
) -> ClipQuality:
    """Decide whether to keep a clip, from its 3D results alone.

    Args:
        left_npz: ``left_results.npz`` -- SOMA-25 joints plus ``cam_t``.
        right_npz: ``right_results.npz``.
        K: ``(3, 3)`` pinhole intrinsics or a camera with a ``project`` method.
            Without it ``root_off_frame`` cannot be tested
            and is reported as zero rather than silently passing.
        image_size: ``(width, height)`` in pixels, needed for the same rule.
        max_bad_frac: share of frame-hands that may be flagged before the clip
            is dropped. **0 means any flagged frame drops the clip**, which is
            the strict default. Raise it to tolerate a fraction of flagged
            predictions; delivery checks still apply regardless of this value.
        fps: source frame rate. Only the speed and acceleration limits use it,
            and they are held per second, so this changes what the same
            displacement means rather than the limit itself.

    Returns:
        A :class:`ClipQuality`. ``drop`` is True when the two hands disagree
        about frame coverage, the delivery is incomplete or contains no finite
        predictions, or the flagged share exceeds ``max_bad_frac``.

    Raises:
        KeyError, ValueError: if a file is not a hand-results npz at all. A
            missing hand is not an error; a malformed file is.
    """
    if looks_like_estimated_K(K, image_size):
        raise ValueError(
            "K is estimate_K's no-intrinsics fallback (focal = max(W, H), principal "
            "point at the image centre), not a calibration. Screening against it "
            "reports a wrong camera as a broken delivery -- on one 1920x1080 "
            "clip it fires root_off_frame on 66% of frame-hands. Pass the real "
            "calibration, e.g. load_intrinsics('undistorted_intrinsics.json')."
        )
    left, lids = _load(left_npz)
    right, rids = _load(right_npz)
    # The clip is the directory holding the two npz files -- the file names are
    # always left_results/right_results and name nothing.
    return check_joints(
        left,
        right,
        left_ids=lids,
        right_ids=rids,
        K=K,
        image_size=image_size,
        max_bad_frac=max_bad_frac,
        fps=fps,
        clip=clip or Path(left_npz).parent.name,
    )


def check_joints(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_ids=None,
    right_ids=None,
    K=None,
    image_size: tuple[int, int] | None = None,
    max_bad_frac: float = 0.0,
    fps: float = 30.0,
    clip: str = "",
    expected_num_frames: int | None = None,
) -> ClipQuality:
    """The same verdict, from joints already in memory.

    ``check_clip`` is a thin loader in front of this. Anything that reads a
    different layout -- a pred_cam export, a parquet shard, a live predictor --
    calls this and gets a decision made by exactly the same code, rather than
    re-implementing the drop logic and drifting from it.

    Args:
        left, right: ``(F, 25, 3)`` camera-frame joints in metres, SOMA-25 order.
            Nonfinite predictions are skipped by geometry checks and counted
            as missing when frame coverage is available.
        left_ids, right_ids: frame ids for each row. Without them the two hands
            can only be compared by row order, and the coverage check -- which
            needs to know what span the clip covers -- is skipped.
        expected_num_frames: total processed video frames, including missing
            detections. When omitted, coverage uses the observed frame span.
            Supply both sets of frame IDs when using this argument.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not 0 <= max_bad_frac <= 1:
        raise ValueError("max_bad_frac must be between 0 and 1")
    if expected_num_frames is not None and (
        expected_num_frames <= 0 or left_ids is None or right_ids is None
    ):
        raise ValueError(
            "expected_num_frames requires a positive count and both frame ID arrays"
        )
    for name, arr in (("left", left), ("right", right)):
        if len(arr) and np.asarray(arr).shape[1:] != (N_JOINTS, 3):
            raise ValueError(
                f"{name}: expected SOMA-25 joints (F,{N_JOINTS},3), "
                f"got {np.asarray(arr).shape}"
            )

    reasons: list[str] = []
    # The two hands must describe the same frames. No threshold, and not
    # tradeable: mismatched rows mean anything pairing them pairs the wrong ones.
    mismatch = len(left) != len(right)
    if mismatch:
        reasons.append(
            f"left has {len(left)} frames, right has {len(right)} -- "
            "the two hands do not cover the same frames"
        )
    elif (
        left_ids is not None
        and right_ids is not None
        and not np.array_equal(left_ids, right_ids)
    ):
        mismatch = True
        reasons.append(
            "left and right cover the same number of frames but not the same frame ids"
        )

    if not any(
        np.isfinite(joints).all(axis=(1, 2)).any()
        for joints in (left, right)
        if len(joints)
    ):
        reasons.append("clip contains no finite hand predictions")
        mismatch = True

    # A hand absent for a large part of the clip is an incomplete delivery, not
    # an occluded one. Frame ids say how much of the span each hand covers;
    # without them the row count is all there is.
    missing: dict[str, int] = {}
    missing_frames: dict[str, list[int]] = {}
    span = expected_num_frames
    first_frame = 0
    if (
        span is None
        and left_ids is not None
        and right_ids is not None
        and len(left_ids)
        and len(right_ids)
    ):
        lo = int(min(np.min(left_ids), np.min(right_ids)))
        hi = int(max(np.max(left_ids), np.max(right_ids)))
        span = hi - lo + 1
        first_frame = lo
    if span is not None and left_ids is not None and right_ids is not None:
        expected_ids = np.arange(first_frame, first_frame + span)
        for side, ids, joints in (
            ("left", left_ids, left),
            ("right", right_ids, right),
        ):
            present_ids = np.asarray(ids)[np.isfinite(joints).all(axis=(1, 2))]
            missing_frames[side] = np.setdiff1d(expected_ids, present_ids).tolist()
            gone = len(missing_frames[side])
            missing[side] = int(gone)
            if span and gone / span > MAX_MISSING_FRAC:
                reasons.append(
                    f"{side} hand is absent from {gone} of {span} frames "
                    f"({gone / span * 100:.1f}%) -- incomplete delivery"
                )
                mismatch = True

    hands = {
        "left": _check_hand(left, K, image_size, fps, "left", left_ids),
        "right": _check_hand(right, K, image_size, fps, "right", right_ids),
    }
    for side, ids in (("left", left_ids), ("right", right_ids)):
        if ids is not None and len(ids) == hands[side].n_frames:
            hands[side].frame_ids = np.asarray(ids)
    for name, idx in _check_pair(left, left_ids, right, right_ids).items():
        for side, rows in idx.items():
            hands[side].flags.setdefault(name, np.zeros(hands[side].n_frames, bool))
            hands[side].flags[name][rows] = True

    total = sum(h.n_frames for h in hands.values())
    bad = sum(int(h.any_flag.sum()) for h in hands.values())
    counts: dict[str, int] = dict.fromkeys(RULES + PAIR_RULES, 0)
    for h in hands.values():
        for k, n in h.counts().items():
            counts[k] = counts.get(k, 0) + n
    frac = bad / total if total else 0.0
    if total and frac > max_bad_frac:
        reasons.append(
            f"{bad}/{total} frame-hands flagged "
            f"({frac * 100:.2f}% > {max_bad_frac * 100:.2f}%)"
        )
    return ClipQuality(
        drop=bool(mismatch or (total and frac > max_bad_frac)),
        reasons=reasons,
        bad_frac=frac,
        n_frames=total,
        hands=hands,
        counts=counts,
        missing=missing,
        missing_frames=missing_frames,
        clip=clip,
    )

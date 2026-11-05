# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Stabilise hand keypoints against the world by correcting the camera pose.

This module is general-purpose -- no dataset-specific assumptions are made.

Problem
-------
Hand keypoints reconstructed per frame in the CAMERA frame look smooth, but once
mapped into world coordinates with SLAM camera poses they vibrate. Real hands do
not vibrate at high frequency, so the jitter is attributed to camera-pose error
and removed by correcting the pose rather than the hands.

Method
------
Alternating optimisation over the poses; the fit itself never touches the
camera-frame keypoints ``X``:

  1. ``W = R X + t``                      current world keypoints
  2. ``S = gaussian_filter(W, sigma)``    smooth target trajectory
  3. per frame, solve ``min ||R'X + t' - S||`` (Kabsch), batched over frames
  4. optionally keep only the fast part of the correction (``sigma_keep``)
  5. optionally smooth what remains (``smooth_post``) -- see below

Steps 1-4 move the camera only, so the keypoints' projection into the image is
bit-for-bit unchanged; they alter how the camera and world frames relate and
cannot fix 2D detection error. Step 5 is different: a rigid camera correction
can only remove jitter common to both hands, and whatever is left is per-hand
noise that only editing the keypoints can remove. It is therefore the one stage
that perturbs the reprojection -- by ~0.15-0.6 px at the default sigma of 1.0,
below the 2D detector's own noise. Set ``smooth_post=0`` to keep the
reprojection exact.

Output frame
------------
Results are returned in the REFINED CAMERA's frame together with the refined
camera-to-world pose; compose the two for world coordinates
(``PoseRefinementResult.to_world()``).

With ``smooth_post=0`` the returned keypoints are numerically identical to the
input and every correction lives in ``poses``. At the default ``smooth_post=1.0``
they carry the post-smoothing as well. How far that moves them is clip
dependent (sub-millimetre on everything measured so far) and is reported per run
as ``metrics['smooth_post_shift_mm']`` rather than assumed. Either way ``poses`` holds the bulk
of the correction, so a consumer that keeps the input keypoints and only swaps
in the refined poses still gets most of the benefit.

Contents of the output npz::

    poses      (F,4,4)   refined camera-to-world  <- use this
    poses_raw  (F,4,4)   the input poses, untouched, for comparison
    left_cam   (F,N,3)   left joints in the refined camera frame, metres
    right_cam  (F,N,3)   right joints, same frame
    inds       (F,)      source frame index, so row order is explicit
    metrics    str       JSON: quality numbers and the params used

Assumption / limits
-------------------
* That the jitter belongs to the camera is an assumption, not a result. It can
  be sanity-checked: if the camera is the cause, both hands must jitter
  together. ``metrics['common_mode_mm']`` vs ``metrics['differential_mm']``
  reports this; if the differential term dominates, the premise does not hold
  for that clip and little jitter will be removable.
* A single rigid transform cannot move the two hands in opposite directions, so
  the differential component is an irreducible floor.
* With ``sigma_keep=0`` (default) the camera is free to move however it must to
  stabilise the hands, so the corrected pose is a fitting parameter and is not
  guaranteed to stay physically plausible. Set ``sigma_keep>0`` (e.g. 15) and/or
  ``mode='translation'`` to keep the trajectory closer to the input.

Usage
-----
As a library::

    from inference.postprocessing import refine_camera_poses_from_npz
    r = refine_camera_poses_from_npz(
            "left_results.npz", "right_results.npz", "pose.npz")
    r.left_cam, r.right_cam, r.poses      # keypoints in the refined camera frame
    left_w, right_w = r.to_world()        # compose with the pose if world is wanted

From the shell::

    python inference/postprocessing/world_coordinate_projection.py \\
        --left left_results.npz \\
        --right right_results.npz --pose pose.npz --out refined_pose.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

__all__ = [
    "PoseRefinementResult",
    "refine_camera_poses",
    "refine_camera_poses_from_npz",
    "load_hand_npz",
    "load_pose_npz",
    "load_pose_inds",
]

PathLike = str | Path


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_hand_npz(path: PathLike) -> np.ndarray:
    """Load one hand's per-frame keypoints in the CAMERA frame, as (F, N, 3).

    Accepts the EgoHand layout ``{joints: (F,N,3) root-relative, cam_t: (F,3)}`` and
    also a plain ``{joints: (F,N,3)}`` that is already camera-frame.
    """
    with np.load(Path(path)) as d:
        if "joints" not in d:
            raise KeyError(f"{path}: expected a 'joints' array, got {list(d.keys())}")
        j = np.asarray(d["joints"], dtype=np.float64)
        if j.ndim != 3 or j.shape[-1] != 3:
            raise ValueError(f"{path}: joints must be (F,N,3), got {j.shape}")
        if "cam_t" in d:  # root-relative -> camera frame
            j = j + np.asarray(d["cam_t"], dtype=np.float64)[:, None, :]
    return j


def load_pose_inds(path: PathLike) -> np.ndarray | None:
    """Frame indices stored alongside the poses, sorted; None if absent."""
    with np.load(Path(path)) as d:
        return np.sort(np.asarray(d["inds"])) if "inds" in d else None


def load_pose_npz(path: PathLike) -> np.ndarray:
    """Load camera-to-world poses as (F, 4, 4).

    Accepts the ViPE layout ``{data: (F,4,4), inds: (F,)}`` (reordered by
    ``inds``) and a bare ``(F,4,4)`` array under ``data``/``poses``/``arr_0``.
    """
    with np.load(Path(path)) as d:
        key = next((k for k in ("data", "poses", "arr_0") if k in d), None)
        if key is None:
            raise KeyError(
                f"{path}: expected 'data'/'poses'/'arr_0', got {list(d.keys())}"
            )
        T = np.asarray(d[key], dtype=np.float64)
        if "inds" in d:
            T = T[np.argsort(np.asarray(d["inds"]))]
    if T.shape[1:] != (4, 4):
        raise ValueError(f"{path}: poses must be (F,4,4), got {T.shape}")
    return T


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(a, sigma, axis=0, mode="nearest")


def _kabsch_batch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame rigid (R, t) minimising ||R P + t - Q||, batched over frames.

    P, Q: (F, N, 3). Uses a batched SVD rather than a Python loop -- identical
    to a per-frame solve up to float roundoff and ~5x faster.
    """
    F = len(P)
    Pm = P.mean(1, keepdims=True)
    Qm = Q.mean(1, keepdims=True)
    H = np.einsum("fki,fkj->fij", P - Pm, Q - Qm)
    U, _, Vt = np.linalg.svd(H)
    # reflection guard: force det(R) = +1
    d = np.sign(np.linalg.det(np.einsum("fij,fkj->fik", Vt, U)))
    D = np.zeros((F, 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("fji,fjk,flk->fil", Vt, D, U)
    return R, Qm[:, 0] - np.einsum("fij,fj->fi", R, Pm[:, 0])


def _to_world(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.einsum("fij,fkj->fki", T[:, :3, :3], X) + T[:, :3, 3][:, None, :]


def refine_camera_poses(
    keypoints_cam: np.ndarray,
    poses_c2w: np.ndarray,
    *,
    sigma_smooth: float = 3.0,
    iters: int = 30,
    sigma_keep: float = 0.0,
    mode: str = "rigid",
    smooth_post: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stabilise world keypoints by correcting the camera poses.

    Parameters
    ----------
    keypoints_cam : (F, N, 3) keypoints in the camera frame. Pass both hands
        stacked along N -- using both at once constrains the fit far better than
        one hand alone.
    poses_c2w : (F, 4, 4) camera-to-world poses.
    sigma_smooth : Gaussian sigma in frames defining "smooth" hand motion.
        Larger removes more jitter but risks flattening genuine fast motion.
    iters : alternating iterations. Converges quickly; 30 is comfortably past it.
    sigma_keep : if > 0, discard pose corrections slower than this many frames,
        so the input's slow trajectory is preserved and only jitter-band energy
        moves into the pose. 0 = unconstrained.
    mode : ``'rigid'`` corrects rotation and translation (best stabilisation);
        ``'translation'`` corrects position only, which keeps the camera more
        plausible but leaves roughly twice the jitter.
    smooth_post : if > 0, Gaussian-smooth the stabilised world track by this
        sigma (frames) as a final pass. Default 1.0, chosen because it keeps
        the induced reprojection shift below one pixel on every clip tested;
        raise it for more smoothing (sigma 2 roughly halves the residual again
        but moves keypoints ~0.5-2 px in the image). The pose correction can only remove
        jitter common to both hands; whatever is left is per-hand noise, and
        only smoothing the keypoints themselves can take it out. UNLIKE the
        pose correction this edits the reconstruction, so the keypoints no
        longer reproject exactly onto the image -- keep sigma small.

    Returns
    -------
    (world, poses) : (F, N, 3) stabilised world keypoints and (F, 4, 4)
        corrected camera-to-world poses. ``world`` equals ``R poses`` applied to
        ``keypoints_cam`` exactly, so the two outputs are consistent.
    """
    X = np.ascontiguousarray(keypoints_cam, dtype=np.float64)
    T0 = np.ascontiguousarray(poses_c2w, dtype=np.float64)
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"keypoints_cam must be (F,N,3), got {X.shape}")
    if T0.shape != (len(X), 4, 4):
        raise ValueError(f"poses must be ({len(X)},4,4), got {T0.shape}")
    if mode not in ("rigid", "translation"):
        raise ValueError(f"mode must be 'rigid' or 'translation', got {mode!r}")
    if X.shape[1] < 3 and mode == "rigid":
        raise ValueError("rigid mode needs >= 3 keypoints per frame")
    # A single non-finite value propagates through the batched SVD and surfaces
    # as an opaque LinAlgError, so check up front and point at the bad frames.
    if not np.isfinite(X).all():
        bad = np.unique(np.argwhere(~np.isfinite(X))[:, 0])
        raise ValueError(
            f"keypoints_cam has non-finite values in {len(bad)} frame(s), "
            f"first at index {bad[0]}; drop or interpolate them first"
        )
    if not np.isfinite(T0).all():
        bad = np.unique(np.argwhere(~np.isfinite(T0))[:, 0])
        raise ValueError(
            f"poses have non-finite values in {len(bad)} frame(s), "
            f"first at index {bad[0]}"
        )

    R0, t0 = T0[:, :3, :3], T0[:, :3, 3]
    W0 = _to_world(X, T0)
    R, t = R0.copy(), t0.copy()

    for _ in range(max(0, iters)):
        W = np.einsum("fij,fkj->fki", R, X) + t[:, None, :]
        S = _smooth(W, sigma_smooth)

        if mode == "rigid":
            Rn, tn = _kabsch_batch(X, S)
        else:  # rotation fixed; centroid offset
            Rn = R0
            tn = S.mean(1) - np.einsum("fij,fj->fi", R0, X.mean(1))

        if sigma_keep > 0:
            # Parameterise the correction about the HAND CENTROID. Anchoring at
            # the world origin lets a rotation and a compensating translation
            # cancel at the hands, so the pose wanders in that null space.
            A = np.einsum("fij,fkj->fik", Rn, R0)
            c0 = W0.mean(1)
            c1 = np.einsum("fij,fj->fi", A, c0 - t0) + tn
            rot = Rotation.from_matrix(A).as_rotvec()
            b = c1 - c0
            A_hp = Rotation.from_rotvec(rot - _smooth(rot, sigma_keep)).as_matrix()
            b_hp = b - _smooth(b, sigma_keep)
            R = A_hp @ R0
            t = c0 + b_hp - np.einsum("fij,fj->fi", A_hp, c0 - t0)
        else:
            R, t = Rn, tn

    T = np.tile(np.eye(4), (len(X), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    W = _to_world(X, T)
    if smooth_post > 0:
        W = _smooth(W, smooth_post)
    return W, T


# --------------------------------------------------------------------------- #
# npz-level API
# --------------------------------------------------------------------------- #
@dataclass
class PoseRefinementResult:
    """Output of :func:`refine_camera_poses_from_npz`. All arrays are float64, metres.

    Keypoints are given in the REFINED CAMERA's frame, paired with the refined
    camera-to-world pose. World coordinates are one composition away::

        W = poses[:, :3, :3] @ X + poses[:, :3, 3]      # see :meth:`to_world`

    Storing it this way keeps the file non-redundant and lets a consumer work in
    whichever frame it needs. The keypoints equal the input exactly when
    ``smooth_post=0``; with post-smoothing on (the default) they also carry that
    edit. See the module docstring on what that costs in reprojection.
    """

    left_cam: np.ndarray  # (F, N, 3) refined-camera frame
    right_cam: np.ndarray  # (F, N, 3) refined-camera frame
    poses: np.ndarray  # (F, 4, 4) refined camera-to-world
    poses_raw: np.ndarray  # (F, 4, 4) input poses, unchanged
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return len(self.poses)

    def to_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Compose the keypoints with the refined pose -> (left, right) in world."""
        R, t = self.poses[:, :3, :3], self.poses[:, :3, 3]
        return (
            np.einsum("fij,fkj->fki", R, self.left_cam) + t[:, None, :],
            np.einsum("fij,fkj->fki", R, self.right_cam) + t[:, None, :],
        )


def _jitter(W: np.ndarray) -> dict[str, float | None]:
    """Jitter descriptors. Entries needing finite differences are None on clips
    too short to form them (accel needs >= 3 frames, step >= 2)."""
    F = len(W)
    acc = np.diff(W, n=2, axis=0) if F >= 3 else None
    step = np.diff(W, axis=0) if F >= 2 else None
    hf = W - _smooth(W, 2.0)
    return {
        "accel_rms_mm": None
        if acc is None
        else float(np.sqrt((acc**2).sum(-1).mean()) * 1000),
        "hf_rms_mm": float(np.sqrt((hf**2).sum(-1).mean()) * 1000),
        "step_median_mm": None
        if step is None
        else float(np.median(np.linalg.norm(step, axis=-1)) * 1000),
    }


def refine_camera_poses_from_npz(
    left_npz: PathLike,
    right_npz: PathLike,
    pose_npz: PathLike,
    *,
    sigma_smooth: float = 3.0,
    iters: int = 30,
    sigma_keep: float = 0.0,
    mode: str = "rigid",
    smooth_post: float = 1.0,
) -> PoseRefinementResult:
    """Stabilise left/right hand keypoints in world coordinates.

    Both hands are stabilised jointly with a single shared camera correction,
    which is the physically meaningful thing to do -- one camera, one pose.

    Parameters mirror :func:`refine_camera_poses`.

    Raises
    ------
    ValueError
        If the two hands and the poses do not have the same number of frames.
    """
    left = load_hand_npz(left_npz)
    right = load_hand_npz(right_npz)
    T0 = load_pose_npz(pose_npz)

    if not (len(left) == len(right) == len(T0)):
        raise ValueError(
            f"frame count mismatch: left={len(left)}, right={len(right)}, poses={len(T0)}"
        )

    nL = left.shape[1]
    X = np.concatenate([left, right], axis=1)

    W0 = _to_world(X, T0)
    W, T = refine_camera_poses(
        X,
        T0,
        sigma_smooth=sigma_smooth,
        iters=iters,
        sigma_keep=sigma_keep,
        mode=mode,
        smooth_post=smooth_post,
    )

    # Express the final world track back in the refined camera frame. With
    # smooth_post=0 this returns the input keypoints exactly (W = T X by
    # construction); with smoothing on it carries the edit.
    X_out = np.einsum("fji,fkj->fki", T[:, :3, :3], W - T[:, :3, 3][:, None, :])

    # Is the premise supported? Camera-caused jitter is common to both hands.
    hf = W0 - _smooth(W0, 2.0)
    mL, mR = hf[:, :nL].mean(1), hf[:, nL:].mean(1)
    dang = np.degrees(
        np.linalg.norm(
            Rotation.from_matrix(
                np.einsum("fij,fkj->fik", T[:, :3, :3], T0[:, :3, :3])
            ).as_rotvec(),
            axis=1,
        )
    )
    metrics = {
        "frames": len(T0),
        "before": _jitter(W0),
        "after": _jitter(W),
        "common_mode_mm": float(np.linalg.norm(((mL + mR) / 2), axis=1).mean() * 1000),
        "differential_mm": float(np.linalg.norm(((mL - mR) / 2), axis=1).mean() * 1000),
        "pose_delta_mm": {
            "median": float(
                np.median(np.linalg.norm(T[:, :3, 3] - T0[:, :3, 3], axis=1)) * 1000
            ),
            "max": float(
                np.max(np.linalg.norm(T[:, :3, 3] - T0[:, :3, 3], axis=1)) * 1000
            ),
        },
        "pose_delta_deg": {"median": float(np.median(dang)), "max": float(dang.max())},
        "params": {
            "sigma_smooth": sigma_smooth,
            "iters": iters,
            "sigma_keep": sigma_keep,
            "mode": mode,
            "smooth_post": smooth_post,
        },
        "smooth_post_shift_mm": {
            "median": float(np.median(np.linalg.norm(X_out - X, axis=-1)) * 1000),
            "max": float(np.max(np.linalg.norm(X_out - X, axis=-1)) * 1000),
        },
    }
    return PoseRefinementResult(X_out[:, :nL], X_out[:, nL:], T, T0, metrics)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stabilise world-frame hand keypoints by correcting camera pose."
    )
    ap.add_argument("--left", required=True, help="left hand npz (joints [+ cam_t])")
    ap.add_argument("--right", required=True, help="right hand npz (joints [+ cam_t])")
    ap.add_argument(
        "--pose", required=True, help="camera-to-world pose npz (data [, inds])"
    )
    ap.add_argument(
        "--out", default=None, help="output npz (default: <pose>_stabilized.npz)"
    )
    ap.add_argument("--sigma_smooth", type=float, default=3.0)
    ap.add_argument(
        "--sigma_keep",
        type=float,
        default=0.0,
        help=">0 keeps the input's slow trajectory; 0 = unconstrained",
    )
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--mode", choices=["rigid", "translation"], default="rigid")
    ap.add_argument(
        "--smooth_post",
        type=float,
        default=1.0,
        help="extra Gaussian smoothing (sigma, frames) of the stabilised "
        "world track; removes the per-hand jitter the pose cannot, "
        "at the cost of editing the keypoints themselves",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    r = refine_camera_poses_from_npz(
        args.left,
        args.right,
        args.pose,
        sigma_smooth=args.sigma_smooth,
        iters=args.iters,
        sigma_keep=args.sigma_keep,
        mode=args.mode,
        smooth_post=args.smooth_post,
    )

    out = (
        Path(args.out)
        if args.out
        else Path(args.pose).with_name(Path(args.pose).stem + "_stabilized.npz")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    inds = load_pose_inds(args.pose)
    if inds is None:
        inds = np.arange(r.frames)
    np.savez(
        out,
        inds=inds,  # frame index, so ordering is explicit
        left_cam=r.left_cam.astype(np.float32),
        right_cam=r.right_cam.astype(np.float32),
        poses=r.poses.astype(np.float32),
        poses_raw=r.poses_raw.astype(np.float32),
        metrics=json.dumps(r.metrics),
    )

    if not args.quiet:
        m = r.metrics
        b, a = m["before"], m["after"]
        print(
            f"frames {m['frames']}  mode={args.mode}  "
            f"sigma_smooth={args.sigma_smooth} sigma_keep={args.sigma_keep}"
        )
        if b["accel_rms_mm"] is None:
            print("  world jitter accel rms : n/a (clip shorter than 3 frames)")
        else:
            print(
                f"  world jitter accel rms : {b['accel_rms_mm']:.3f} -> "
                f"{a['accel_rms_mm']:.3f} mm  "
                f"({100 * (a['accel_rms_mm'] - b['accel_rms_mm']) / b['accel_rms_mm']:+.1f}%)"
            )
        print(
            f"  world jitter hf rms    : {b['hf_rms_mm']:.3f} -> {a['hf_rms_mm']:.3f} mm"
        )
        print(
            f"  jitter split           : common-mode {m['common_mode_mm']:.3f} mm "
            f"vs differential {m['differential_mm']:.3f} mm "
            f"(differential is the irreducible floor)"
        )
        print(
            f"  pose correction        : {m['pose_delta_mm']['median']:.2f} mm median, "
            f"{m['pose_delta_deg']['median']:.3f} deg median"
        )
        print(f"  saved -> {out}")
        print(
            "  keys: left_cam/right_cam (refined camera frame), poses "
            "(refined c2w), poses_raw, metrics;  world = poses @ cam"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

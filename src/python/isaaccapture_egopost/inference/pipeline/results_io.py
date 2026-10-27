# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Writers for EgoHand inference results (per-frame and per-clip)."""

from pathlib import Path

import numpy as np


KEYPOINT_RELIABILITY_FIELDS = ("pred_kp2d_crop_presence", "pred_kp2d_sigma")


def save_hand_results(out_dir, frame_names, results):
    """Save per-frame SOMA params + 3D outputs to .npz files."""
    results_dir = Path(out_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, r in zip(frame_names, results):
        save_kwargs = dict(
            hand_poses=r["hand_poses"],  # (25, 3, 3) rot matrices
            identity_coeffs=r["identity_coeffs"],  # (20,) SOMA or (5,) MHR
            global_scale=np.array(r["global_scale"]),
            vertices=r["pred_vertices"],  # (Vh, 3) meters, wrist-local
            joints=r["pred_keypoints_3d"],  # (25, 3) meters, wrist-local
            cam_t=r["pred_cam_t"],  # (3,) camera translation
        )
        # Producers disagree on how they signal "no per-finger scale head":
        # run_hand_inference above omits the key, while the predictor in
        # inference/pipeline/predictor.py always sets it and leaves it None.
        # `in r` alone accepts the latter and writes an object array that only
        # loads back under allow_pickle=True.
        if r.get("scale_params") is not None:
            save_kwargs["scale_params"] = r["scale_params"]  # (26,) MHR only
        for key in KEYPOINT_RELIABILITY_FIELDS:
            if r.get(key) is not None:
                save_kwargs[key] = np.asarray(r[key], dtype=np.float32)
        np.savez(results_dir / f"{name}.npz", **save_kwargs)

    print(f"[HandModel] Saved {len(results)} result files to {results_dir}/")


def save_hand_results_clip(out_path, frame_names, results):
    """Save a whole clip's SOMA params + 3D outputs to a single .npz file.

    Same fields as :func:`save_hand_results`, but with every per-frame array
    stacked along a leading frame axis of length N.

    N is the number of frames that carried a box for *this* hand, which is NOT
    the clip length and generally differs between the left and right files:
    frames the detector left without a box are simply absent. Read
    ``frame_names[i]`` to map row ``i`` back to its source frame -- consumers
    must not index these arrays by frame number -- and branch on ``num_frames``
    before indexing, since a hand that never got a box yields only ``num_frames=0``
    and an empty ``frame_names`` array.

    Args:
        out_path: Destination ``.npz`` path.
        frame_names: Frame IDs, one per entry of ``results``, same order.
        results: Per-frame result dicts in ``save_hand_results`` schema.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_names = [str(name) for name in frame_names]
    if len(frame_names) != len(results):
        raise ValueError(
            f"frame_names/results length mismatch: {len(frame_names)} vs {len(results)}"
        )

    # A hand can go boxless for the entire clip (single-hand footage, or a
    # detector that never fires). That is a legitimate result, not an error, so
    # still write a file -- consumers globbing for <hand>_results.npz should not
    # have to distinguish "absent" from "empty".
    if not results:
        np.savez(
            out_path,
            num_frames=np.array(0),
            frame_names=np.array([], dtype="<U1"),
        )
        print(f"[HandModel] No frames detected -- wrote empty clip file to {out_path}")
        return

    save_kwargs = dict(
        num_frames=np.array(len(results)),
        frame_names=np.array(frame_names),
        hand_poses=np.stack([r["hand_poses"] for r in results]),  # (N, 25, 3, 3)
        identity_coeffs=np.stack(
            [r["identity_coeffs"] for r in results]
        ),  # (N, 20) or (N, 5)
        global_scale=np.array([r["global_scale"] for r in results]),  # (N,)
        vertices=np.stack([r["pred_vertices"] for r in results]),  # (N, Vh, 3)
        joints=np.stack([r["pred_keypoints_3d"] for r in results]),  # (N, 25, 3)
        cam_t=np.stack([r["pred_cam_t"] for r in results]),  # (N, 3)
    )

    # scale_params is None for checkpoints without a per-finger scale head. It
    # is all-or-nothing per checkpoint, so a mixed clip means something upstream
    # is wrong -- say so rather than silently dropping the field.
    n_scale = sum(r.get("scale_params") is not None for r in results)
    if n_scale == len(results):
        save_kwargs["scale_params"] = np.stack(
            [r["scale_params"] for r in results]
        )  # (N, 26)
    elif n_scale:
        print(
            f"[HandModel] WARNING: scale_params present on only {n_scale}/{len(results)} "
            f"frames; omitting it from {out_path.name}"
        )

    for key in KEYPOINT_RELIABILITY_FIELDS:
        values = [r.get(key) for r in results]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(
                f"{key} must be present for every result or absent for all"
            )
        save_kwargs[key] = np.stack(values).astype(np.float32)

    np.savez(out_path, **save_kwargs)
    print(f"[HandModel] Saved {len(results)} frames to {out_path}")

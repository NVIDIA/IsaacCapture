# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Reproducible native parameter export and optional image overlays."""

from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from .data import save_json, sha256_file
from .visualization import write_skeleton_overlay
from .fitting import tensor
from .losses import preservation_report


@torch.no_grad()
def export_results(models, states, data, args, outdir, reports):
    import soma

    runtime_root = Path(soma.__file__).resolve().parent
    outdir = Path(outdir)
    T = data.nframes
    if len({m.nshape for m in models}) != 1 or len({m.nscale for m in models}) != 1:
        raise ValueError("Both hands must use matching identity parameter dimensions.")
    beta = np.full((T, 2, models[0].nshape), np.nan, np.float32)
    rotations = np.full((T, 2, 25, 3, 3), np.nan, np.float32)
    poses = np.full((T, 2, 25, 3), np.nan, np.float32)
    translations = np.full((T, 2, 3), np.nan, np.float32)
    scales = np.full((T, 2), np.nan, np.float32)
    bones = np.full((T, 2, models[0].nscale), np.nan, np.float32)
    joints = np.full((T, 2, 25, 3), np.nan, np.float32)
    observations = np.full((T, 2, 21, 3), np.nan, np.float32)
    latent = np.full((T, 2, models[0].npose), np.nan, np.float32)
    valid = np.stack([s.valid for s in states], axis=1)
    warnings = np.stack([s.failed for s in states], axis=1)
    for side, (m, s) in enumerate(zip(models, states)):
        m.prepare_identity(
            tensor(s.beta, args.device),
            tensor(s.log_scale, args.device),
            tensor(s.bone_logs, args.device) if args.optimize_bones else None,
        )
        ids = np.flatnonzero(s.valid)
        for start in tqdm(range(0, len(ids), 128), desc=f"Export {m.side}"):
            chunk = ids[start : start + 128]
            _, j, q, R = m.pose_prepared(
                tensor(s.latent[chunk], args.device),
                tensor(s.root[chunk], args.device),
                tensor(s.translation[chunk], args.device),
            )
            rr, jj, qq = R.cpu().numpy(), j.cpu().numpy(), q.cpu().numpy()
            finite = np.isfinite(rr).all((1, 2, 3)) & np.isfinite(jj).all((1, 2))
            valid[chunk[~finite], side] = False
            warnings[chunk[~finite], side] = True
            keep = chunk[finite]
            if not len(keep):
                continue
            beta[keep, side] = s.beta[0]
            rotations[keep, side] = rr[finite]
            poses[keep, side] = (
                Rotation.from_matrix(rr[finite].reshape(-1, 3, 3))
                .as_rotvec()
                .reshape(-1, 25, 3)
            )
            translations[keep, side] = s.translation[keep]
            scales[keep, side] = np.exp(s.log_scale[0])
            if m.identity_backend == "mhr":
                bones[keep, side] = s.bone_logs[0] if args.optimize_bones else 0
            else:
                bones[keep, side] = np.exp(s.bone_logs[0]) if args.optimize_bones else 1
            joints[keep, side] = jj[finite]
            observations[keep, side] = qq[finite]
            latent[keep, side] = s.latent[keep]
    path = outdir / "soma_parameters.npz"
    np.savez_compressed(
        path,
        identity_coeffs=beta,
        poses=poses,
        rotation_matrices=rotations,
        global_translation=translations,
        global_scale=scales,
        scale_params=bones,
        scale_params_convention=np.array(
            "mhr_additive"
            if models[0].identity_backend == "mhr"
            else "soma_multiplicative"
        ),
        native_root_joint_index=np.array(0),
        input_wrist_camera=data.points[:, :, 0],
        wrist_correspondence_verified=np.array(False),
        joints25=joints,
        observations21=observations,
        observation_joint_indices=np.stack([m.joint_map.cpu().numpy() for m in models]),
        observation_local_offsets_m=np.stack([m.offsets.cpu().numpy() for m in models]),
        observation_weights=np.stack(
            [m.observation_weights.cpu().numpy() for m in models]
        ),
        pose_latent=latent,
        pose_mean=np.stack([m.mean.cpu().numpy() for m in models]),
        pose_basis=np.stack([m.basis.cpu().numpy() for m in models]),
        pose_parameterization=np.array(
            "native_pca_plus_orthogonal_residual"
            if models[0].npose > 32
            else "native_pca"
        ),
        valid=valid,
        numerical_warning=warnings,
        timestamps=data.timestamps,
        frame_index=np.arange(T),
        hand_sides=np.array(["left", "right"]),
        joint_names=np.array([m.joint_names for m in models]),
        joint_parent_ids=np.stack(
            [m.layer.joint_parent_ids.cpu().numpy() for m in models]
        ),
        K=data.K,
        absolute_pose=np.array(True),
        identity_model_type=np.array(models[0].identity_backend),
        units=np.array("m"),
        coordinate_system=np.array("opencv_camera"),
        lod=np.array(args.lod),
    )
    metric_report = preservation_report(
        observations, joints[:, :, 0], translations, data.points, data.confidence, valid
    )
    save_json(outdir / "metric_preservation.json", metric_report)
    versions = {}
    for package in (
        "py-soma-x",
        "torch",
        "numpy",
        "opencv-python-headless",
        "transformers",
        "triton",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    save_json(
        outdir / "report.json",
        {
            "parameters": path.name,
            "versions": versions,
            "native_asset_sha256": {
                n: sha256_file(Path(args.assets) / n) for n in models[0].asset_files
            },
            "configuration": vars(args),
            "native_runtime": "py-soma-x",
            "native_runtime_sha256": {
                "soma/" + p.relative_to(runtime_root).as_posix(): sha256_file(p)
                for p in sorted(runtime_root.rglob("*.py"))
            },
            "input_sha256": {
                "video": sha256_file(data.video),
                "points": sha256_file(data.points_file),
            },
            "source_sha256": {
                p.name: sha256_file(p)
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "observation_mapping_sha256": sha256_file(args.mapping)
            if args.mapping
            else None,
            "hands": dict(zip(("left", "right"), reports)),
            "metric_preservation": metric_report,
            "conversion_contract": {
                "root": "global_translation equals native joints25[...,0,:] in meters in the OpenCV camera frame",
                "wrist_correspondence": {
                    "source_point_index": 0,
                    "native_joint_index": 0,
                    "mode": args.wrist_source,
                    "hard_radius_m": args.wrist_radius
                    if args.wrist_source == "landmark"
                    else None,
                    "anatomical_correspondence_verified": False,
                    "note": "The landmark mode assumes input point 0 and native root 0 denote the same joint center. A small numerical root residual does not verify that assumption or a downstream conversion.",
                },
                "scale": "Pass global_scale and native scale_params through the native layer exactly once; do not normalize mesh size or root-center camera-space outputs",
                "scale_parameter_names": {
                    m.side: list(
                        getattr(m.layer.identity_model, "scale_param_names", [])
                    )
                    for m in models
                },
                "evaluation": "A downstream model conversion must explicitly preserve metric size and define its root convention; this fit does not run that conversion",
            },
            "valid_frames_per_hand": valid.sum(0).tolist(),
            "scope": "Reference-free native hand fitting using py-soma-x. No external wrist calibration or MANO assets are used.",
            "limitations": [
                "Masks cannot identify hidden geometry or guarantee metric scale.",
                "Shared identity assumes one person and one persistent track per side.",
                "Unmodeled object occlusion must be excluded using ignore masks or clean frames.",
                "Articulation is regularized by the native pose prior; unobserved joint twists remain ambiguous.",
            ],
        },
    )
    if args.overlay:
        parent_ids = np.stack([m.layer.joint_parent_ids.cpu().numpy() for m in models])
        write_skeleton_overlay(data, joints, valid, parent_ids, outdir)
    return path

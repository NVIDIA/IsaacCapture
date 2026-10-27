# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Command-line entry points for native assets and full video fitting."""

import argparse
import logging
from pathlib import Path
import random
import time
import json
import numpy as np
import torch
from .data import load_inputs, sha256_file, save_json
from .model import NativeHand, fetch_assets
from .preprocess import prepare, PREPROCESSING_VERSION, preprocessing_signature
from .render import SilhouetteRenderer
from .fitting import fit_side
from .export import export_results

logger = logging.getLogger(__name__)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    assets = sub.add_parser("assets", help="Download native SOMAHand identity assets")
    assets.add_argument("--out", required=True)
    assets.add_argument("--revision", default="v0.3.0")
    assets.add_argument("--identity-backend", choices=["soma", "mhr"], default="mhr")
    fit = sub.add_parser("fit", help="Fit video and frame-aligned 3D landmarks")
    fit.add_argument("--video", required=True)
    fit.add_argument("--points", required=True)
    fit.add_argument("--assets", required=True)
    fit.add_argument("--identity-backend", choices=["soma", "mhr"], default="mhr")
    fit.add_argument(
        "--out",
        default="output_run",
        help="New or empty output directory (default: output_run)",
    )
    fit.add_argument(
        "--device", default="cuda:0", help="Torch device (default: cuda:0)"
    )
    fit.add_argument("--renderer", choices=["auto", "soft", "triton"], default="triton")
    fit.add_argument(
        "--mask-backend", choices=["sam3", "files", "grabcut"], default="grabcut"
    )
    fit.add_argument("--mask-dir")
    fit.add_argument(
        "--reuse-observations",
        help="Reuse a matching observations directory from an earlier run",
    )
    fit.add_argument("--ignore-dir")
    fit.add_argument(
        "--sam3-checkpoint",
        default="facebook/sam3",
        help="Hugging Face SAM3 model ID or local save_pretrained directory",
    )
    fit.add_argument(
        "--mapping",
        help="Semantic indices/weights only; calibration offsets are rejected",
    )
    fit.add_argument(
        "--wrist-source",
        choices=["palm", "landmark"],
        default="palm",
        help="Infer native wrist from palm geometry, or trust a convention-matched wrist landmark",
    )
    fit.add_argument(
        "--wrist-radius",
        type=float,
        default=0.001,
        help="Maximum native wrist displacement from a valid input wrist in meters; -1 disables",
    )
    fit.add_argument(
        "--metric-size-weight",
        type=float,
        default=1.0,
        help="Preserve robust input palm/finger dimensions during fitting; 0 disables",
    )
    fit.add_argument(
        "--metric-shrink-weight",
        type=float,
        default=3.0,
        help="Additional penalty on undersized palm/finger dimensions",
    )
    fit.add_argument(
        "--metric-size-tolerance",
        type=float,
        default=0.02,
        help="Relative dead zone for metric hand dimensions",
    )
    fit.add_argument(
        "--metric-shape-steps",
        type=int,
        default=100,
        help="Geometry-only shape initialization before image fitting",
    )
    fit.add_argument("--reprojection-weight", type=float, default=0.25)
    fit.add_argument("--reprojection-tolerance", type=float, default=2.0, help="Pixels")
    fit.add_argument("--reprojection-scale", type=float, default=10.0, help="Pixels")
    fit.add_argument(
        "--holdout-limit",
        type=int,
        default=16,
        help="Maximum diagnostic holdout frames per hand",
    )
    fit.add_argument("--lod", choices=["low", "mid"], default="low")
    fit.add_argument("--image-size", type=int, default=96)
    fit.add_argument(
        "--pose-components",
        type=int,
        default=72,
        help="Up to 32 native PCA modes; 33-72 add orthogonal native joint-rotation residuals",
    )
    fit.add_argument("--keyframes", type=int, default=8)
    fit.add_argument("--residual-candidates", type=int, default=12)
    fit.add_argument("--add-keyframes", type=int, default=2)
    fit.add_argument("--batch-size", type=int, default=8)
    fit.add_argument("--keypoint-batch-size", type=int, default=128)
    fit.add_argument(
        "--sequence-batch-size",
        type=int,
        default=64,
        help="Renderer batch size for final sequence fitting; objective and temporal links are unchanged",
    )
    fit.add_argument(
        "--observation-host-cache-mib",
        type=int,
        default=256,
        help="Host observation cache per hand; increase for long episodes without changing fitting",
    )
    fit.add_argument(
        "--observation-device-cache-mib",
        type=int,
        default=128,
        help="Immutable observation tensor cache on GPU; 0 disables",
    )
    fit.add_argument(
        "--sequence-polish-steps",
        type=int,
        default=20,
        help="Final native pose refinement at one tenth learning rate; 0 disables",
    )
    fit.add_argument("--init-steps", type=int, default=80)
    fit.add_argument("--pose-steps", type=int, default=20)
    fit.add_argument("--shape-steps", type=int, default=40)
    fit.add_argument("--final-steps", type=int, default=100)
    fit.add_argument("--rounds", type=int, default=2)
    fit.add_argument("--pose-lr", type=float, default=0.025)
    fit.add_argument("--shape-lr", type=float, default=0.01)
    fit.add_argument("--translation-lr", type=float, default=0.001)
    fit.add_argument("--landmark-tolerance", type=float, default=0.001)
    fit.add_argument("--landmark-scale", type=float, default=0.01)
    fit.add_argument("--keypoint-weight", type=float, default=12.0)
    fit.add_argument("--image-weight", type=float, default=1.0)
    fit.add_argument(
        "--wrist-surface-weight",
        type=float,
        default=2.0,
        help="Balanced visible wrist-band surface loss, independent of finger/palm area",
    )
    fit.add_argument(
        "--surface-coverage-weight",
        type=float,
        default=2.0,
        help="Penalty for visible target hand area missed by the native surface",
    )
    fit.add_argument(
        "--shape-visibility-fraction",
        type=float,
        default=0.8,
        help="Shared shape uses views near the clearest finger visibility; 0 disables",
    )
    fit.add_argument(
        "--surface-refine-steps",
        type=int,
        default=30,
        help="Late native shape refinement after sequence pose alignment; 0 disables",
    )
    fit.add_argument("--shape-prior", type=float, default=0.01)
    fit.add_argument("--pose-prior", type=float, default=0.003)
    fit.add_argument("--scale-prior", type=float, default=0.02)
    fit.add_argument("--bone-prior", type=float, default=0.003)
    fit.add_argument("--scale-mode", choices=["metric", "fixed"], default="metric")
    fit.add_argument("--initial-scale", type=float, default=1.0)
    fit.add_argument(
        "--mhr-scale-limit",
        type=float,
        default=2.0,
        help="Absolute limit for native additive MHR scale parameters (not a fractional length)",
    )
    fit.add_argument(
        "--optimize-bones",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimize native bone scale controls (default: enabled)",
    )
    fit.add_argument("--temporal-weight", type=float, default=1.0)
    fit.add_argument(
        "--temporal-target",
        choices=["smooth_observations", "observations", "stationary"],
        default="smooth_observations",
        help="Track bounded smooth observation motion; raw residual and stationary modes remain available",
    )
    fit.add_argument(
        "--temporal-smoothing-window",
        type=float,
        default=0.067,
        help="Temporal target regression radius in seconds; 0 disables target smoothing",
    )
    fit.add_argument(
        "--temporal-max-correction",
        type=float,
        default=0.002,
        help="Maximum temporal target displacement from each raw observation, in metres",
    )
    fit.add_argument(
        "--temporal-mesh-jerk-weight",
        type=float,
        default=0.05,
        help="Native surface jerk regularization, including motion not determined by keypoints",
    )
    fit.add_argument(
        "--temporal-root-jerk-weight",
        type=float,
        default=0.1,
        help="Native wrist trajectory jerk regularization, independent of source root convention",
    )
    fit.add_argument("--temporal-max-gap", type=float, default=0.1)
    fit.add_argument(
        "--overlay",
        action="store_true",
        help="Generate fitted SOMA 25-joint skeleton overlay.mp4 (default: disabled)",
    )
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--cpu-threads", type=int, default=4)
    fit.add_argument("--preprocess-workers", type=int, default=4)
    compare = sub.add_parser(
        "compare", help="Compare input 21-point and fitted SOMA 25-joint skeletons"
    )
    compare.add_argument("--video", required=True)
    compare.add_argument("--points", required=True)
    compare.add_argument("--fit", required=True, help="Exported soma_parameters.npz")
    compare.add_argument("--out", required=True, help="New comparison MP4 path")
    return p


def validate_args(a):
    for name in (
        "batch_size",
        "keypoint_batch_size",
        "sequence_batch_size",
        "observation_host_cache_mib",
        "keyframes",
        "residual_candidates",
        "pose_components",
        "cpu_threads",
        "preprocess_workers",
    ):
        if getattr(a, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if not 1 <= a.pose_components <= 72:
        raise ValueError("pose_components must be in [1,72].")
    if a.image_size < 32 or a.image_size % 8:
        raise ValueError("image_size must be a multiple of 8 and >=32.")
    for name in (
        "observation_device_cache_mib",
        "init_steps",
        "sequence_polish_steps",
        "pose_steps",
        "shape_steps",
        "final_steps",
        "rounds",
        "add_keyframes",
        "holdout_limit",
        "metric_shape_steps",
        "surface_refine_steps",
    ):
        if getattr(a, name) < 0:
            raise ValueError(f"{name} must be nonnegative.")
    for name in (
        "pose_lr",
        "shape_lr",
        "translation_lr",
        "landmark_scale",
        "reprojection_scale",
        "initial_scale",
        "mhr_scale_limit",
        "metric_shrink_weight",
    ):
        if not np.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    for name in (
        "landmark_tolerance",
        "metric_size_weight",
        "metric_size_tolerance",
        "reprojection_tolerance",
        "reprojection_weight",
        "keypoint_weight",
        "image_weight",
        "surface_coverage_weight",
        "wrist_surface_weight",
        "shape_visibility_fraction",
        "shape_prior",
        "pose_prior",
        "scale_prior",
        "bone_prior",
        "temporal_weight",
        "temporal_smoothing_window",
        "temporal_max_correction",
        "temporal_mesh_jerk_weight",
        "temporal_root_jerk_weight",
        "temporal_max_gap",
    ):
        if not np.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not np.isfinite(a.wrist_radius) or (a.wrist_radius < 0 and a.wrist_radius != -1):
        raise ValueError("wrist_radius must be nonnegative or -1 to disable.")
    if a.shape_visibility_fraction > 1:
        raise ValueError("shape_visibility_fraction must be in [0,1].")
    if a.metric_size_tolerance >= 1:
        raise ValueError("metric_size_tolerance must be less than 1.")
    if str(a.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu --renderer soft.")


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "assets":
        print(fetch_assets(args.out, args.revision, args.identity_backend))
        return
    if args.command == "compare":
        from .visualization import write_comparison

        write_comparison(args.video, args.points, args.fit, args.out)
        return
    validate_args(args)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Output directory is nonempty. Choose a new directory to prevent accidental overwrites."
        )
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    data = load_inputs(args.video, args.points)
    models = [
        NativeHand(
            args.assets,
            side,
            args.device,
            args.lod,
            args.pose_components,
            args.mapping,
            args.identity_backend,
            args.wrist_source,
        )
        for side in ("left", "right")
    ]
    renderer = SilhouetteRenderer(args.renderer, args.device, args.image_size)
    logger.info(
        "Renderer: %s; SOMAHand identity backend: %s; no MANO assets used",
        renderer.backend,
        args.identity_backend,
    )
    args.renderer = renderer.backend

    def timestamp():
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
        return time.perf_counter()

    model_ready = timestamp()
    timings = {"input_and_models_seconds": model_ready - started}
    cache = out / "observations"
    if args.reuse_observations:
        cache = Path(args.reuse_observations)
        provenance = json.loads((cache / "provenance.json").read_text())
        expected = {
            "video_sha256": sha256_file(data.video),
            "points_sha256": sha256_file(data.points_file),
            "image_size": args.image_size,
            "frame_count": data.nframes,
            "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessing_source_sha256": preprocessing_signature(),
            "mask_seed": args.seed,
        }
        if any(provenance.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "Observation cache does not match inputs, image size, seed, or current preprocessing code; rebuild observations."
            )
        with np.load(cache / "index.npz", allow_pickle=False) as d:
            quality, descriptors, usable = (
                np.array(d[k]) for k in ("quality", "descriptors", "usable")
            )
        logger.info("Reusing cached masks and trust regions from %s", cache)
    else:
        quality, descriptors, usable = prepare(data, cache, args)
    observations_ready = timestamp()
    timings["observations_seconds"] = observations_ready - model_ready
    states, reports = [], []
    for side, model in enumerate(models):
        fit_started = timestamp()
        state, report = fit_side(
            model,
            renderer,
            data,
            side,
            quality[:, side],
            descriptors[:, side],
            usable[:, side],
            cache,
            args,
            out,
        )
        timings[f"{model.side}_fit_seconds"] = timestamp() - fit_started
        states.append(state)
        reports.append(report)
    export_started = timestamp()
    result = export_results(models, states, data, args, out, reports)
    timings["export_and_overlay_seconds"] = timestamp() - export_started
    timings["total_seconds"] = timestamp() - started
    save_json(out / "timing_breakdown.json", timings)
    print(f"Saved parameters: {result}")


if __name__ == "__main__":
    main()

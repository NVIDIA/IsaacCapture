# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""End-to-end EgoHand hand inference on video files.

Runs 2D hand bbox detection then EgoHandPredictor inference directly on
decoded video frames with a EgoHand checkpoint.

Usage (single video):
    python inference/scripts/run_egohand_inference_pipeline.py \\
        --input /path/to/video.mp4 \\
        --output_dir outputs/infer_video/ \\
        --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml \\
        --draw_skeleton --no_mesh --dtype bfloat16

Usage (folder of MP4s):
    python inference/scripts/run_egohand_inference_pipeline.py \\
        --input /path/to/videos/ \\
        --output_dir outputs/infer_video/ \\
        --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml \\
        --draw_skeleton --no_mesh --dtype bfloat16

Usage (with the camera calibration instead of the focal heuristic):
    python inference/scripts/run_egohand_inference_pipeline.py \\
        --input /path/to/video.mp4 \\
        --output_dir outputs/infer_video/ \\
        --intrinsics_json /path/to/intrinsics.json \\
        --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml \\
        --perspective_cropping --draw_skeleton --no_mesh --dtype bfloat16

Every example above omits --egohand_ckpt to use the default HF-downloaded
checkpoint (see --egohand_ckpt's help); pass it explicitly to use a local one.

Metric output depends on the intrinsics in every camera mode: the predicted
camera is converted to a translation with ``compute_transl_full_cam(pred_cam,
bbx_xys, K)``, so depth scales with ``fx``. Without ``--intrinsics_json``,
this falls back to ``estimate_K``'s ``focal = max(W, H)``, which puts every
hand at the wrong distance whenever the true focal differs from that
heuristic -- invisibly, because the overlay renders through the same wrong
``K``. Checkpoints trained with ``camera_mode: sam3db_perspective`` also feed
``K`` to the network, so a wrong one perturbs the prediction as well; those get
an explicit warning when ``--intrinsics_json`` is omitted.
"""

# ruff: noqa: E402
import argparse
import functools
import json
import os
import sys
from pathlib import Path
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch

torch.load = functools.partial(torch.load, weights_only=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference.camera.types import ImageData
from inference.pipeline.predictor import EgoHandPredictor
from inference.pipeline.pipeline import EgoHandInferencePipeline
from inference.postprocessing import check_hand_results
from inference.postprocessing.clip_quality_control import looks_like_estimated_K
from inference.assets import EGOHAND_HF_REPO, hf_download
from inference.camera.intrinsics import resolve_K
from inference.camera.square_pixel import (
    SquarePixelResizeParams,
    compute_square_pixel_resize_params,
    rewrite_bbox_json_in_original_frame,
)
from inference.preprocessing.crop import crop_and_preprocess
from inference.hand_detector_2d.hand_bbox_detector import SquarePixelFrameBBoxDetector

# Detector backends (rtdetr, saved_bboxes) are imported dynamically by name
# in _build_detector_from_config -- this script never imports a detector
# class directly, only via the capability-checked HandBBoxDetector Protocol
# (prepare_clip / writes_bbox_json / run).
from inference.perf import LocalPerfWriter, PerfStage
from inference.pipeline.results_io import save_hand_results, save_hand_results_clip
from inference.visualization.overlay import render_clip

FPS_FALLBACK = 30.0
MESH_COLOR = (1.0, 1.0, 0.9)


def get_video_meta(video_path: str) -> dict:
    """Return fps, width, height, num_frames without decoding any frames."""
    cap = cv2.VideoCapture(video_path)
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def read_video_frames_batched(
    video_path: str,
    batch_size: int,
    max_frames: int | None = None,
):
    """Generator that yields lists of (frame_idx, BGR_frame), batch_size at a time.

    Each yielded list has exactly batch_size entries, except possibly the last
    which may be smaller. Stops early when max_frames is reached.
    """
    cap = cv2.VideoCapture(video_path)
    idx = 0
    batch = []
    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            batch.append((idx, frame))
            idx += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        cap.release()


def _load_detector_config(path: str) -> dict:
    """Load a bbox detector YAML config, expanding ${ENV_VAR} references."""
    import os
    import yaml

    with open(path) as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)


def _build_detector_from_config(cfg: dict, device: str, batch_size: int) -> tuple:
    """Build (bbox_detector, backend) from a loaded detector config dict.

    Dispatches to ``inference.hand_detector_2d.configs.<detector_backend>.build``
    by name -- this script never imports a detector backend module or class
    directly, only the naming convention. Add a backend by adding
    ``inference/hand_detector_2d/configs/<name>.py`` with a
    ``build(cfg, device, batch_size, tracking)`` function; nothing here needs to
    change.
    """
    import importlib

    backend = cfg["detector_backend"]
    tracking = cfg.get("tracking", False)
    module_name = f"inference.hand_detector_2d.configs.{backend}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(f"Unknown detector_backend: {backend!r}") from exc
        raise
    return module.build(cfg, device, batch_size, tracking), backend


def parse_args():
    p = argparse.ArgumentParser(
        description="E2E EgoHand hand inference with 2D hand bbox detection on video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, help="MP4 file or folder of MP4 files")
    p.add_argument(
        "--egohand_ckpt",
        default=None,
        help="EgoHand checkpoint (.ckpt). Default: downloaded from "
        "the nvidia/EgoHand-v1 HF repo (see --egohand_ckpt_hf_path) and cached locally -- "
        "pass an explicit path to use a local checkpoint instead.",
    )
    p.add_argument(
        "--egohand_ckpt_hf_path",
        default="egohand_s060000.ckpt",
        help="Filename within the nvidia/EgoHand-v1 HF repo to fetch for --egohand_ckpt's "
        "default. Only consulted when --egohand_ckpt is not given.",
    )
    p.add_argument(
        "--bbox_detector_config",
        default=None,
        metavar="YAML",
        help="Path to a bbox detector config YAML (inference/hand_detector_2d/configs/). "
        "Supports ${ENV_VAR} substitution for per-video paths. "
        "Replaces the former --detector/--sam3_ckpt/--vitpose_ckpt/etc. flags.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Output root; per-video results go into <output_dir>/<video_stem>/",
    )
    p.add_argument(
        "--soma_hand_assets",
        default=None,
        help="Directory containing SOMAHand.npz, SOMA_neutral.npz, SOMA_template_rig.usda, "
        "SOMA_procedural_transforms.json, and MHR/{mhr_model_lod1.pt, base_body_lod1.obj, "
        "SOMA_wrap_lod1.obj} (the default checkpoint uses the MHR identity model). "
        "Default: resolved via soma.get_assets_dir(), which downloads the public asset "
        "snapshot from https://huggingface.co/nvidia/SOMA-X on first use and caches it "
        "locally (see the py-soma-x package). Pass this to use a local copy instead.",
    )
    p.add_argument(
        "--intrinsics_json",
        default=None,
        help="Shared camera JSON (model, image_width, image_height, "
        "camera_matrix.matrix_3x3, distortion_coefficients); see README. "
        "Use the decoded video's resolution/orientation. Default: None "
        "(estimate focal length from image dimensions).",
    )
    p.add_argument(
        "--perspective_cropping",
        action="store_true",
        default=False,
        help="Use the calibrated source model for virtual pinhole hand crops. "
        "Requires --intrinsics_json. Default: off (undistort full frames, then crop hands).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for EgoHand (default: 32). RT-DETR takes its own "
        "via the batch_size: key in the rtdetr_config YAML mapping.",
    )
    p.add_argument(
        "--dino_compile",
        action="store_true",
        help="Compile each DINO block for eval/no-grad full batches of "
        "--batch_size. Training, grad-enabled calls, and partial batches use "
        "the eager path. Use for throughput-sensitive batch inference where "
        "most batches are full-size (e.g. long videos); skip it for small "
        "one-off runs, since torch.compile warm-up cost may exceed the "
        "savings. Off by default, so it has no effect unless passed.",
    )
    p.add_argument(
        "--postprocessing_smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adaptively smooth SOMA hand rotations, translation and shape using "
        "video timestamps, then rebuild geometry and re-project 2D keypoints. "
        "Filtering spans the complete clip, including across --frame_batch_size "
        "boundaries. See inference/postprocessing/adaptive_smoothing.py. "
        "Distinct from bbox tracking "
        "(the tracking: key in --bbox_detector_config), which post-processes the 2D "
        "bboxes going into the model rather than the predictions coming out. "
        "Default: on. Pass --no-postprocessing_smooth to disable.",
    )
    p.add_argument(
        "--postprocessing_qc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run 3D quality control once per complete video and write quality_control.json. "
        "Predictions and overlays are preserved even when QC reports drop=true. "
        "Requires --hand both and camera intrinsics. Default: off.",
    )
    p.add_argument(
        "--qc_max_bad_frac",
        type=float,
        default=0.02,
        help="Allowed fraction of flagged frame-hands for QC, in [0, 1]. "
        "Default: 0.02 (up to 2%% flagged frame-hands). Coverage checks always apply.",
    )
    p.add_argument("--hand", choices=["left", "right", "both"], default="both")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Inference dtype (default: float32)",
    )
    p.add_argument(
        "--rectify_pixel_aspect_ratio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For calibrations with fx != fy, correct non-square camera pixels by "
        "resizing each frame and adjusting its intrinsics before detection and "
        "inference. Overlays, bbox JSONs and the 3D outputs are all expressed in "
        "the original frame. For PINHOLE input, per-frame NPZs from --save_npzs "
        "retain their corrected-frame 2D fields. "
        "No effect with --perspective_cropping "
        "or when fx == fy. Default: on. Pass --no-rectify_pixel_aspect_ratio "
        "to disable.",
    )
    p.add_argument("--no_save_videos", action="store_true", help="Skip .mp4 outputs")
    p.add_argument(
        "--no_mesh",
        action="store_true",
        help="Skip mesh rendering in overlays (skeleton only if --draw_skeleton). "
        "Enabled automatically for non-PINHOLE input as rendering hand mesh "
        "in distorted fisheye camera space is unsupported.",
    )
    p.add_argument(
        "--save_crop_overlays",
        action="store_true",
        help="Save crop_overlay_<hand>.mp4 (and crop_overlay_<hand>/ images). "
        "Off by default -- bbox_viz.mp4 and fullimg_overlay.mp4 are unaffected "
        "either way. Crop overlays are mainly a debugging aid for bbox/crop "
        "alignment: the crop window is centered on each frame's raw detected "
        "bbox, which --postprocessing_smooth never touches, so the crop can "
        "look jittery even though the mesh/pose inside it is smoothed like "
        "everywhere else. Disabled with --perspective_cropping or non-PINHOLE input.",
    )
    p.add_argument(
        "--save_images",
        action="store_true",
        help="Save per-frame .jpg outputs (default: off)",
    )
    p.add_argument(
        "--save_npzs",
        action="store_true",
        help="Save per-frame full SOMA .npz outputs (default: off).",
    )
    p.add_argument(
        "--per_frame_results",
        action="store_true",
        help="Write one result .npz per frame under <video>/<hand>/results/ (legacy layout). "
        "Default is a single <video>/<hand>_results.npz per hand for the whole clip.",
    )
    p.add_argument(
        "--perf_output",
        default=None,
        help="Write perf metrics as JSON to this path "
        "(default: <output_dir>/<video_stem>/perf.json for each video)",
    )
    p.add_argument(
        "--max_seconds",
        type=float,
        default=None,
        help="Limit inference to the first N seconds of each video (default: full video).",
    )
    p.add_argument(
        "--draw_skeleton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the 2D joint skeleton overlay (default: on). "
        "Pass --no-draw-skeleton to disable.",
    )
    p.add_argument(
        "--frame_batch_size",
        type=int,
        default=None,
        help="Stream video in batches of this many frames to cap peak RAM. "
        "Disables video/image saving. Recommended for long videos (default: disabled).",
    )
    args = p.parse_args()
    if args.frame_batch_size is not None and args.frame_batch_size < 1:
        p.error("--frame_batch_size must be positive")
    if not 0 <= args.qc_max_bad_frac <= 1:
        p.error("--qc_max_bad_frac must be between 0 and 1")
    if args.postprocessing_qc:
        if args.hand != "both":
            p.error("--postprocessing_qc requires --hand both")
        if not args.intrinsics_json:
            p.error("--postprocessing_qc requires --intrinsics_json")
    return args


def find_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.mp4"))


def process_video(
    video_path,
    pipeline,
    hands,
    output_dir,
    K_fullimg,
    save_images,
    save_videos,
    write_soma_npzs,
    per_frame_results=False,
    perf_reporter: LocalPerfWriter = None,
    no_mesh=False,
    frame_batch_size=None,
    draw_skeleton=True,
    max_seconds=None,
    resize_params: SquarePixelResizeParams | None = None,
    save_crop_overlays=False,
    postprocessing_qc=False,
    qc_max_bad_frac=0.02,
    undistort_maps=None,
    source_camera=None,
):
    """Run detection, inference and rendering for one video.

    ``postprocessing_qc`` evaluates both hands after all batches are collected
    and writes ``quality_control.json`` without changing any result exports.
    It is disabled by default; ``qc_max_bad_frac`` controls its strictness.

    ``K_fullimg`` describes the full frames: a pinhole matrix (after any
    undistortion and square-pixel correction), or a ``CameraModel`` object for perspective
    cropping. With perspective cropping, the model receives the crop's virtual
    pinhole intrinsics.

    With ``resize_params`` the clip-level outputs undo only pixel-aspect resizing
    -- the 3D is frame-independent, overlays are resized at write time, and
    bbox JSONs are mapped on the way out. For PINHOLE input, per-frame NPZs written
    by ``write_soma_npzs`` are the exception: their ``pred_keypoints_2d`` is
    normalized relative to the crop in the corrected image.
    """
    total_wall_start = time.perf_counter()
    smooth_bboxes = getattr(pipeline.bbox_detector, "requires_full_clip", False)
    if smooth_bboxes and frame_batch_size is not None:
        raise ValueError(
            "bbox_smoothing requires whole-clip detection before EgoHand cropping. "
            "Omit --frame_batch_size, or generate boxes with run_hand_detector.py "
            "and use saved_bboxes.yaml for batched EgoHand inference. "
            "Setting bbox_smoothing: false permits batching, but live RT-DETR "
            "tracking then runs independently within each batch."
        )
    print(f"\n[infer_video] === {video_path.name} ===")

    meta = get_video_meta(str(video_path))
    W, H = meta["width"], meta["height"]
    fps = meta["fps"] or FPS_FALLBACK
    if not np.isfinite(fps) or fps <= 0:
        fps = FPS_FALLBACK
    num_frames = meta["num_frames"]
    max_frames = int(max_seconds * fps) if max_seconds is not None else None

    if resize_params is not None and resize_params.is_identity:
        resize_params = None
    # Past this point every pixel quantity -- frames, boxes, K, overlays -- is
    # in the model frame; only what is written out is mapped back to W x H.
    W_model, H_model = (
        (resize_params.width, resize_params.height) if resize_params else (W, H)
    )
    if resize_params is not None:
        print(
            f"[infer_video] pixel-aspect rectify {W}x{H} -> {W_model}x{H_model} "
            f"(sx={resize_params.scale_x:.4f}, sy={resize_params.scale_y:.4f}); "
            f"outputs written in the original {W}x{H} frame"
        )

    if postprocessing_qc:
        if set(hands) != {"left", "right"}:
            raise ValueError("QC requires both left and right hands")
        if not 0 <= qc_max_bad_frac <= 1:
            raise ValueError("qc_max_bad_frac must be between 0 and 1")
        if looks_like_estimated_K(K_fullimg, (W_model, H_model)):
            raise ValueError(
                "QC requires real camera intrinsics, not the estimate_K fallback"
            )

    out_dir = output_dir / video_path.stem
    report_path = out_dir / "quality_control.json"
    # This run replaces the outputs. A disabled or interrupted QC run must
    # not leave an older verdict attached to the new predictions.
    report_path.unlink(missing_ok=True)

    # Bbox JSONs are appended to frame batch by frame batch below, so clear any
    # existing bbox JSON for this video first to start fresh. Only if the detector
    # will actually write one: writes_bbox_json is a required accessor on every
    # HandBBoxDetector (e.g. saved_bboxes is False -- it only reads from
    # --bbox-dir and never writes) -- capability-checked, not special-cased by
    # class or defaulted if missing.
    if pipeline.bbox_detector is not None and pipeline.bbox_detector.writes_bbox_json:
        for hand in hands:
            (out_dir / f"{video_path.stem}_{hand}_bbox.json").unlink(missing_ok=True)

    # Stream frames in batches to cap peak RAM. When frame_batch_size is None,
    # a single batch holds the full video (original behaviour).
    batch_size = frame_batch_size or 10**9
    inference_results: dict[str, dict] = {"left": {}, "right": {}}
    smooth = pipeline.predictor.smooth
    defer_smoothing = frame_batch_size is not None and smooth
    # Batched inference releases images; smoothing metadata cannot enable rendering.
    render_overlays = (save_images or save_videos) and frame_batch_size is None
    clip_frame_times = {}
    clip_bboxes = {}
    clip_intrinsics = {}
    source_images, source_boxes = {}, {}
    total_frames = 0

    print(
        f"[infer_video] running pipeline (bbox detection + EgoHand): batch size {frame_batch_size}, total frames: {num_frames} ..."
    )
    for batch in read_video_frames_batched(str(video_path), batch_size, max_frames):
        print(f"[infer_video] running batch with size {len(batch)}")
        if perf_reporter is not None:
            perf_reporter.record_frames(len(batch))

        if source_camera is not None and render_overlays:
            source_images.update((f"{idx:06d}", frame) for idx, frame in batch)
        if undistort_maps is not None or resize_params is not None:
            # Replace each frame in place so peak memory does not double.
            for i in range(len(batch)):
                idx, frame = batch[i]
                if undistort_maps is not None:
                    frame = cv2.remap(
                        frame, *undistort_maps, interpolation=cv2.INTER_LINEAR
                    )
                if resize_params is not None:
                    frame = resize_params.apply_to_frame(frame)
                batch[i] = (idx, frame)

        image_data_batch = {
            f"{idx:06d}": ImageData(
                bgr=frame,
                intrinsics=K_fullimg,
                camera_to_world_transform=None,
                gt_2d_bboxes=None,
            )
            for idx, frame in batch
        }
        smoothing_options = {}
        if smooth or smooth_bboxes:
            smoothing_options["frame_times"] = {
                f"{idx:06d}": idx / fps for idx, _ in batch
            }
        if defer_smoothing:
            clip_frame_times.update(smoothing_options["frame_times"])
            smoothing_options["defer_smoothing"] = True
        result = pipeline.run(
            image_data_per_image=image_data_batch,
            return_debug_data=render_overlays,
            write_soma_npzs=write_soma_npzs and not defer_smoothing,
            output_folder=out_dir,
            **smoothing_options,
            source_camera=source_camera,
        )
        if defer_smoothing:
            clip_bboxes.update(result["bboxes_per_image"])
            clip_intrinsics.update(result["intrinsics_per_image"])
        if source_camera is not None:
            source_boxes.update(result["bboxes_per_image"])
        for h in hands:
            inference_results[h].update(result["inference_results_per_hand"][h])
        total_frames += len(batch)
        del image_data_batch, batch

    if total_frames == 0:
        raise RuntimeError(f"[infer_video] {video_path}: no frames decoded.")
    if defer_smoothing:
        pipeline.predictor.postprocess_results(
            inference_results,
            list(clip_frame_times),
            clip_bboxes,
            frame_times=clip_frame_times,
            intrinsics_per_image=clip_intrinsics,
            source_cameras_per_image=(
                dict.fromkeys(clip_frame_times, source_camera)
                if source_camera is not None
                else None
            ),
        )
        if write_soma_npzs:
            pipeline.predictor.save_results(
                inference_results,
                {
                    hand: {
                        image_id: out_dir / f"frame{image_id}_{hand}_soma.npz"
                        for image_id in records
                    }
                    for hand, records in inference_results.items()
                },
            )
    if max_frames is not None:
        print(
            f"[infer_video] trimmed to first {max_seconds:.1f}s ({total_frames} frames)"
        )

    print(f"[infer_video] {total_frames} frames, {W}x{H} @ {fps:.1f} fps")

    for h in hands:
        n_inferred = len(inference_results[h])
        n_missing = total_frames - n_inferred
        print(
            f"[infer_video] {h}: {n_inferred}/{total_frames} frames inferred "
            f"({n_missing} skipped — no prediction)"
        )

    if postprocessing_qc:
        verdict = check_hand_results(
            inference_results,
            num_frames=total_frames,
            K=source_camera if source_camera is not None else K_fullimg,
            image_size=(W, H) if source_camera is not None else (W_model, H_model),
            fps=fps,
            max_bad_frac=qc_max_bad_frac,
            clip=video_path.stem,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(verdict.to_json(with_frames=True, indent=2) + "\n")
        print(f"[infer_video] QC: {verdict.summary()}")
        print(
            f"[infer_video] QC report written to {report_path}; outputs are preserved"
        )

    if not any(inference_results[h] for h in hands):
        raise RuntimeError(
            f"[infer_video] {video_path}: 0/{total_frames} frames inferred across "
            "the requested hands. Check bbox inputs, frame IDs, and detector confidence."
        )

    for h, results in inference_results.items():
        remapped = [
            {**r, "hand_poses": r["poses"], "pred_cam_t": r["transl"]}
            for r in results.values()
        ]
        if per_frame_results:
            save_hand_results(out_dir / h, list(results.keys()), remapped)
        else:
            save_hand_results_clip(
                out_dir / f"{h}_results.npz", list(results.keys()), remapped
            )

    if render_overlays:
        print("[infer_video] rendering overlays ...")
        bboxes_per_image = result["bboxes_per_image"]
        faces_per_hand = {
            h: (f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f))
            for h, f in result["faces_per_hand"].items()
        }
        images_bgr = (
            source_images if source_camera is not None else result["images_bgr"]
        )

        # Use all frames; None = no detection for that hand on that frame.
        valid_fids = list(images_bgr)
        valid_frames = [images_bgr[fid] for fid in valid_fids]

        results_demo = {}
        for h in hands:
            results_demo[h] = []
            for fid in valid_fids:
                if fid in inference_results.get(h, {}):
                    r = inference_results[h][fid]
                    demo_r = dict(r)
                    demo_r["pred_cam_t"] = r["transl"]
                    demo_r["hand_poses"] = r["poses"]
                    results_demo[h].append(demo_r)
                else:
                    results_demo[h].append(None)

        bbx_list = {
            h: [
                bboxes_per_image[fid].get(h) if fid in bboxes_per_image else None
                for fid in valid_fids
            ]
            for h in hands
        }
        crops_per_hand = {}
        for h in hands:
            crops, bbxs_viz, Ks_viz = [], [], []
            for frame, fid in zip(valid_frames, valid_fids):
                bbx = bboxes_per_image.get(fid, {}).get(h)
                if bbx is not None:
                    _, crop_rgb = crop_and_preprocess(frame, bbx)
                    crops.append(crop_rgb)
                    bbxs_viz.append(bbx)
                else:
                    crops.append(None)
                    bbxs_viz.append(None)
                Ks_viz.append(K_fullimg)
            crops_per_hand[h] = (crops, bbxs_viz, Ks_viz)

        overlay_start = time.perf_counter()
        render_clip(
            out_dir,
            valid_frames,
            valid_fids,
            results_demo,
            crops_per_hand,
            bbx_list,
            faces_per_hand,
            hands,
            W if source_camera is not None else W_model,
            H if source_camera is not None else H_model,
            source_camera if source_camera is not None else K_fullimg,
            fps,
            MESH_COLOR,
            save_images,
            save_videos,
            draw_skeleton=draw_skeleton,
            no_mesh=no_mesh or source_camera is not None,
            display_size=(W, H)
            if resize_params is not None and source_camera is None
            else None,
            save_crop_overlays=save_crop_overlays and source_camera is None,
        )
        if perf_reporter is not None and valid_fids:
            overlay_ms_per_frame = (
                (time.perf_counter() - overlay_start) * 1000 / len(valid_fids)
            )
            perf_reporter.record_latency(
                PerfStage.OVERLAY_RENDERING, overlay_ms_per_frame
            )

    if source_camera is not None:
        for hand in hands:
            boxes = {
                fid: value[hand].tolist()
                for fid, value in source_boxes.items()
                if hand in value
            }
            (out_dir / f"{video_path.stem}_{hand}_bbox.json").write_text(
                json.dumps(boxes)
            )
    else:
        rewrite_bbox_json_in_original_frame(out_dir, hands, resize_params)

    if perf_reporter is not None and total_frames:
        total_wall_ms_per_frame = (
            (time.perf_counter() - total_wall_start) * 1000 / total_frames
        )
        perf_reporter.record_latency(PerfStage.TOTAL_WALL, total_wall_ms_per_frame)

    print(f"[infer_video] done → {out_dir}")


def main():
    args = parse_args()
    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_path)
    if not videos:
        raise SystemExit(f"No MP4 files found at {input_path}")
    print(f"[infer_video] {len(videos)} video(s) to process")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Read calibration once, then validate each video's camera
    # before downloading weights or constructing the detector and predictor.
    parsed_calibration = None
    video_cameras = {}
    use_perspective_cropping = args.perspective_cropping
    try:
        if args.intrinsics_json:
            with Path(args.intrinsics_json).open() as calibration_file:
                parsed_calibration = json.load(calibration_file)
        for video_path in videos:
            video_metadata = get_video_meta(str(video_path))
            source_camera, K_source = resolve_K(
                video_metadata["width"],
                video_metadata["height"],
                parsed_calibration,
                perspective_cropping=use_perspective_cropping,
                calibration_source=(
                    f"calibration {args.intrinsics_json}"
                    if args.intrinsics_json
                    else None
                ),
            )
            video_cameras[video_path] = (source_camera, K_source, video_metadata)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Invalid camera calibration: {exc}") from None

    if use_perspective_cropping:
        args.save_crop_overlays = False

    if not args.bbox_detector_config:
        raise ValueError(
            "--bbox_detector_config is required. See inference/hand_detector_2d/configs/."
        )

    _detector_cfg = _load_detector_config(args.bbox_detector_config)
    bbox_detector, backend = _build_detector_from_config(
        _detector_cfg, args.device, args.batch_size
    )
    if hasattr(bbox_detector, "validate_inputs"):
        bbox_detector.validate_inputs(videos, hands)
    if (
        not use_perspective_cropping
        and getattr(bbox_detector, "reports_in_source_frame", False)
        and any(
            not isinstance(camera[0], np.ndarray) for camera in video_cameras.values()
        )
    ):
        raise ValueError(
            "Full-image undistortion requires detection on the undistorted frames. "
            "Use a live bbox detector, or --perspective_cropping with source-frame boxes."
        )

    # Resolved after the argument/detector checks above so an invalid
    # combination fails immediately instead of after the multi-GB download.
    if args.egohand_ckpt is None:
        args.egohand_ckpt = hf_download(args.egohand_ckpt_hf_path)
        print(
            f"[infer_video] egohand_ckpt not given, resolved via HF ({EGOHAND_HF_REPO}/"
            f"{args.egohand_ckpt_hf_path}): {args.egohand_ckpt}"
        )

    if args.soma_hand_assets is None:
        from soma import get_assets_dir

        args.soma_hand_assets = str(get_assets_dir())
        print(
            f"[infer_video] soma_hand_assets not given, resolved via soma.get_assets_dir(): "
            f"{args.soma_hand_assets}"
        )

    print(
        f"[infer_video] loading EgoHand model + {backend} detector (config: {args.bbox_detector_config}) ..."
    )
    t0 = time.perf_counter()
    predictor = EgoHandPredictor.from_checkpoint(
        egohand_ckpt_path=args.egohand_ckpt,
        soma_hand_data_folder=args.soma_hand_assets,
        batch_size=args.batch_size,
        device=args.device,
        dtype=dtype,
        smooth=args.postprocessing_smooth,
        use_perspective_cropping=use_perspective_cropping,
    )
    if args.dino_compile:
        predictor.enable_dino_compile()
        print(
            "[infer_video] DINO per-block compilation enabled: "
            f"full batch {args.batch_size}; training, gradients, and partial "
            "batches use eager"
        )
    model_load_ms = (time.perf_counter() - t0) * 1000

    pipeline = EgoHandInferencePipeline(
        predictor=predictor, bbox_detector=bbox_detector
    )

    save_images = args.save_images
    save_videos = not args.no_save_videos
    write_soma_npzs = args.save_npzs

    camera_mode = getattr(predictor.model, "camera_mode", "weak_perspective")
    print(f"[infer_video] checkpoint camera_mode={camera_mode}")
    if args.postprocessing_smooth:
        print(
            "[infer_video] adaptive SOMA smoothing on: rotations, translation and "
            "shape; geometry rebuilt from filtered parameters"
        )
    if camera_mode != "weak_perspective" and not args.intrinsics_json:
        print(
            "[infer_video] WARNING: no --intrinsics_json was given; falling back to "
            "an estimated focal length (max(W, H)). The result will not be accurate "
            "-- using real camera intrinsics is strongly recommended."
        )

    for video_path in videos:
        # An explicit path collects the whole run; otherwise report each video.
        if not args.perf_output or video_path == videos[0]:
            perf_output_path = (
                Path(args.perf_output)
                if args.perf_output
                else output_dir / video_path.stem / "perf.json"
            )
            perf_reporter: LocalPerfWriter = LocalPerfWriter(
                output_path=perf_output_path,
                metadata={
                    "checkpoint": Path(args.egohand_ckpt).stem,
                    "device": (
                        torch.cuda.get_device_name(args.device)
                        if torch.cuda.is_available()
                        else "unknown"
                    ),
                    "batch_size": args.batch_size,
                    "dtype": args.dtype,
                    "dino_compile": args.dino_compile,
                    "perspective_cropping": use_perspective_cropping,
                },
            )
            perf_reporter.record_latency(PerfStage.MODEL_LOAD, model_load_ms)
            pipeline.perf_reporter = perf_reporter
            predictor.perf_reporter = perf_reporter

        K_original, K_source, video_metadata = video_cameras[video_path]
        W, H = video_metadata["width"], video_metadata["height"]
        if (
            not args.no_mesh
            and not isinstance(K_original, np.ndarray)
            and K_original.model_name != "PINHOLE"
        ):
            print(
                "[infer_video] Mesh rendering skipped: not supported for "
                "non-pinhole cameras."
            )
        undistort_maps = None
        if not use_perspective_cropping and not isinstance(K_original, np.ndarray):
            source_camera = K_original
            # Retain resolution and K values, now defining a pinhole output image.
            # Wide-angle content outside this pinhole field of view is cropped.
            K_original = source_camera.K.astype(np.float32)
            map_xy = np.empty((H, W, 2), dtype=np.float32)
            # Limit temporary projection/Jacobian storage by constructing row strips.
            for top in range(0, H, 128):
                yy, xx = np.mgrid[top : min(top + 128, H), :W]
                rays = np.stack(
                    (
                        (xx - K_original[0, 2]) / K_original[0, 0],
                        (yy - K_original[1, 2]) / K_original[1, 1],
                        np.ones_like(xx),
                    ),
                    axis=-1,
                )
                pixels = source_camera.project(rays)
                map_xy[top : top + 128] = np.where(
                    np.isfinite(pixels).all(axis=-1, keepdims=True), pixels, -1
                )
            undistort_maps = cv2.convertMaps(map_xy, None, cv2.CV_16SC2)
            del map_xy
            print(
                f"[infer_video] Full-image undistortion: {source_camera.model_name} -> PINHOLE"
            )
        if use_perspective_cropping:
            print(
                f"[infer_video] Perspective cropping with {K_original.model_name}, {W}x{H} "
                f"({K_source})"
            )
        else:
            print(
                f"[infer_video] K_fullimg fx={K_original[0, 0]:.2f} fy={K_original[1, 1]:.2f} "
                f"cx={K_original[0, 2]:.2f} cy={K_original[1, 2]:.2f} ({K_source})"
            )

        # K_model is what the model actually sees; it differs from the source
        # camera only when the pixel aspect ratio is rectified.
        K_model = K_original
        resize_params = None
        if args.rectify_pixel_aspect_ratio and not use_perspective_cropping:
            params = compute_square_pixel_resize_params(K_original, W, H)
            if params.is_identity:
                print(
                    "[infer_video] --rectify_pixel_aspect_ratio: pixels already square, no-op"
                )
            else:
                resize_params, K_model = params, params.K
                print(
                    f"[infer_video] K_model    fx={K_model[0, 0]:.2f} "
                    f"fy={K_model[1, 1]:.2f} cx={K_model[0, 2]:.2f} "
                    f"cy={K_model[1, 2]:.2f} ({params.width}x{params.height})"
                )

        # Configure per-video detector state before inference through one
        # capability-probed hook rather than by backend name.
        # Only genuinely per-clip values go here (video_path, K); anything constant
        # across the whole run (e.g. saved_bboxes' bbox_dir root) is bound once at
        # construction time in the detector's own configs/<backend>.py builder.
        # K_original, not K_model: a detector that projects ground truth does so
        # into the source frame, and reports_in_source_frame boxes are mapped
        # forward below.
        if hasattr(bbox_detector, "prepare_clip"):
            bbox_detector.prepare_clip(video_path=video_path, K=K_original)

        # Boxes that came from a file describe the source frame; boxes from a
        # detector that looked at the frames already describe the corrected one.
        pipeline.bbox_detector = (
            SquarePixelFrameBBoxDetector(bbox_detector, resize_params)
            if resize_params is not None
            and getattr(bbox_detector, "reports_in_source_frame", False)
            else bbox_detector
        )

        process_video(
            video_path,
            pipeline,
            hands,
            output_dir,
            K_model,
            save_images,
            save_videos,
            write_soma_npzs,
            per_frame_results=args.per_frame_results,
            perf_reporter=perf_reporter,
            frame_batch_size=args.frame_batch_size,
            draw_skeleton=args.draw_skeleton,
            no_mesh=args.no_mesh,
            max_seconds=args.max_seconds,
            resize_params=resize_params,
            save_crop_overlays=args.save_crop_overlays,
            postprocessing_qc=args.postprocessing_qc,
            qc_max_bad_frac=args.qc_max_bad_frac,
            undistort_maps=undistort_maps,
            source_camera=source_camera if undistort_maps is not None else None,
        )

        if not args.perf_output or video_path == videos[-1]:
            out = perf_reporter.write()
            print(f"[infer_video] perf results written to {out}")

    print(f"\n[infer_video] all done. results in {output_dir}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Argument parsing without model imports, including runtime capability checks."""

import argparse
import os

from inference.media.runtime_profile import (
    configure_parser,
    get_runtime_profile,
    validate_args,
)


def _bundled_default(env_var):
    """Path named by ``env_var`` if that file exists, else None (flag stays unset).

    Lets an image bundle a default checkpoint / detector config: the image sets the
    variable, and an explicit flag or a mounted replacement file still wins.
    """
    path = os.environ.get(env_var)
    return path if path and os.path.isfile(path) else None


def parse_args(argv=None, *, epilog: str | None = None):
    p = argparse.ArgumentParser(
        description="E2E EgoHand hand inference with 2D hand bbox detection on video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("--input", required=True, help="MP4 file or folder of MP4 files")
    p.add_argument(
        "--egohand_ckpt",
        default=_bundled_default("EGOHAND_DEFAULT_CKPT"),
        help="EgoHand checkpoint (.ckpt). Default: the file named by "
        "$EGOHAND_DEFAULT_CKPT if it exists (bundled in the release image), otherwise downloaded from "
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
        default=_bundled_default("EGOHAND_DEFAULT_DETECTOR_CONFIG"),
        metavar="YAML",
        help="Path to a bbox detector config YAML (inference/hand_detector_2d/configs/). "
        "Default: the file named by $EGOHAND_DEFAULT_DETECTOR_CONFIG if it exists. "
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
    try:
        profile = get_runtime_profile()
    except ValueError as exc:
        p.error(str(exc))
    configure_parser(p, profile)
    args = p.parse_args(argv)
    validate_args(p, args, profile)
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

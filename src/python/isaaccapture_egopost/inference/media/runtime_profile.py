# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Runtime capabilities shared by EgoHand and the standalone video reader.

EGOHAND_RUNTIME_PROFILE describes installed capabilities, rather than a per-video
preference. Container builds set it to release; source installs default to full.
"""

import argparse
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    video_backend: str
    rendering: bool
    compilation: bool


_PROFILES = {
    "full": RuntimeProfile("full", "opencv", rendering=True, compilation=True),
    "release": RuntimeProfile("release", "ffmpeg", rendering=False, compilation=False),
}


def get_runtime_profile() -> RuntimeProfile:
    name = os.environ.get("EGOHAND_RUNTIME_PROFILE", "full")
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown EGOHAND_RUNTIME_PROFILE={name!r}; choose full or release"
        ) from exc


# Help for the release image, which bundles the checkpoint, detector weights and SOMA
# assets, never renders, and rejects --frame_batch_size with whole-clip bbox smoothing.
_RELEASE_HELP = {
    "bbox_detector_config": (
        "Bbox detector config YAML. Default: $EGOHAND_DEFAULT_DETECTOR_CONFIG "
        "(/workspace/artifacts/rtdetr.yaml), which uses the bundled RT-DETR weights. "
        "The configs under inference/hand_detector_2d/configs/ are source files that "
        "do not point at the bundled weights; to change a setting, edit a copy of the "
        "default config and pass it here. Supports ${ENV_VAR} substitution."
    ),
    "soma_hand_assets": (
        "Directory containing SOMAHand.npz, SOMA_neutral.npz, SOMA_template_rig.usda, "
        "SOMA_procedural_transforms.json, and MHR/{mhr_model_lod1.pt, base_body_lod1.obj, "
        "SOMA_wrap_lod1.obj}. Default: the asset snapshot bundled in the image, so no "
        "download is needed. Pass this to use a local copy instead."
    ),
    "intrinsics_json": (
        "Shared camera JSON (model, image_width, image_height, "
        "camera_matrix.matrix_3x3, distortion_coefficients); an example is at the "
        "end of this help. Use the decoded video's resolution/orientation. "
        "Default: None (estimate focal length from image dimensions)."
    ),
    "dtype": "Inference dtype (default: float32; the examples below use bfloat16)",
    "postprocessing_qc": (
        "Run 3D quality control once per complete video and write quality_control.json. "
        "Predictions are preserved even when QC reports drop=true. "
        "Requires --hand both and camera intrinsics. Default: off."
    ),
    "rectify_pixel_aspect_ratio": (
        "For calibrations with fx != fy, correct non-square camera pixels by "
        "resizing each frame and adjusting its intrinsics before detection and "
        "inference. Bbox JSONs and the 3D outputs are all expressed in the "
        "original frame. For PINHOLE input, per-frame NPZs from --save_npzs "
        "retain their corrected-frame 2D fields. "
        "No effect with --perspective_cropping "
        "or when fx == fy. Default: on. Pass --no-rectify_pixel_aspect_ratio "
        "to disable."
    ),
    "no_save_videos": (
        "Accepted for compatibility; this image never writes .mp4 files."
    ),
    "frame_batch_size": (
        "Stream video in batches of this many frames to cap peak RAM. Recommended "
        "for long videos (default: disabled). The default detector config smooths "
        "boxes over the whole clip (bbox_smoothing: true), which this flag rejects: "
        "use saved boxes (detector_backend: saved_bboxes) or set bbox_smoothing: "
        "false in a copy of the config. See the long-video example below."
    ),
}

RELEASE_EPILOG = r"""
Usage (single video; the checkpoint, detector weights and SOMA assets are bundled):
    python3 inference/scripts/run_egohand_inference_pipeline.py \
        --input /path/to/video.mp4 \
        --intrinsics_json /path/to/intrinsics.json \
        --perspective_cropping --dtype bfloat16 \
        --output_dir /out

Usage (folder of MP4s; all videos must come from the camera in --intrinsics_json):
    python3 inference/scripts/run_egohand_inference_pipeline.py \
        --input /path/to/videos/ --intrinsics_json /path/to/intrinsics.json \
        --perspective_cropping --dtype bfloat16 --output_dir /out

Long videos (--frame_batch_size) need saved boxes. Detect them first, then run
EgoHand on them with a config containing
    detector_backend: saved_bboxes
    saved_bbox_path: /path/to/boxes
which is the --output_dir of the detector run:
    python3 inference/hand_detector_2d/scripts/run_hand_detector.py \
        --input /path/to/long.mp4 --output_dir /path/to/boxes \
        --bbox_detector_config /workspace/artifacts/rtdetr.yaml \
        --intrinsics_json /path/to/intrinsics.json --frame_batch_size 128
    python3 inference/scripts/run_egohand_inference_pipeline.py \
        --input /path/to/long.mp4 --intrinsics_json /path/to/intrinsics.json \
        --bbox_detector_config /path/to/saved_bboxes.yaml \
        --frame_batch_size 128 --perspective_cropping --output_dir /out

Every example above uses the bundled checkpoint and detector; pass --egohand_ckpt
or --bbox_detector_config to use your own.

Camera JSON for a pinhole camera (other models: KB4, FISHEYE624,
RATIONAL_POLYNOMIAL; use --perspective_cropping for fisheye and
rational-polynomial cameras):
    {"model": "PINHOLE", "image_width": 1920, "image_height": 1080,
     "camera_matrix": {"matrix_3x3": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0],
                                      [0.0, 0.0, 1.0]]},
     "distortion_coefficients": []}

Metric output depends on the intrinsics in every camera mode: depth scales with
``fx``. Without ``--intrinsics_json``, the focal length is estimated as
``max(W, H)``, which puts every hand at the wrong distance whenever the true
focal differs, and nothing in the outputs shows it. Checkpoints trained with
``camera_mode: sam3db_perspective`` also feed ``K`` to the network, so a wrong
one perturbs the prediction as well; those get an explicit warning when
``--intrinsics_json`` is omitted.
"""


def configure_parser(parser: argparse.ArgumentParser, profile: RuntimeProfile) -> None:
    parser.description += f" (runtime profile: {profile.name})"
    unavailable = set()
    if not profile.compilation:
        unavailable.add("dino_compile")
    if not profile.rendering:
        unavailable.update(("save_images", "save_crop_overlays", "draw_skeleton"))
    release_help = _RELEASE_HELP if profile.name == "release" else {}
    if release_help:
        parser.epilog = RELEASE_EPILOG
    for action in parser._actions:
        if action.dest in unavailable:
            action.help = argparse.SUPPRESS
        elif action.dest in release_help:
            action.help = release_help[action.dest]
        elif not profile.rendering and action.dest == "no_mesh":
            action.help = (
                "Accepted for compatibility; mesh rendering is already disabled."
            )
    if not profile.rendering:
        parser.set_defaults(no_save_videos=True, no_mesh=True, draw_skeleton=False)


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, profile: RuntimeProfile
) -> None:
    if not profile.compilation and args.dino_compile:
        parser.error("--dino_compile is unsupported")
    if not profile.rendering:
        for option in ("save_images", "save_crop_overlays", "draw_skeleton"):
            if getattr(args, option):
                parser.error(f"--{option} is unsupported")

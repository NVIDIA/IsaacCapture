# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Stand-alone RT-DETRv2 hand bbox detection: video in, per-hand bbox JSON out.

This is the detection half of ``run_egohand_inference_pipeline.py`` pulled out into
its own entrypoint, so bbox detection can be run and reviewed as a separate
preprocessing stage instead of an in-process flag. The JSONs it writes are
exactly what ``saved_bbox.SavedBBoxDetector`` reads, so the second pass is

    <output_dir>/<video_stem>/<video_stem>_{left,right}_bbox.json

and the pipeline consumes it with
``--bbox_detector_config inference/hand_detector_2d/configs/saved_bboxes.yaml``
(that config's ``saved_bbox_path`` is ``${BBOX_DIR}``, i.e. the ``--output_dir``
used here; the pipeline appends the video stem itself).

The standalone detector and live pipeline detection accept the same RT-DETR
config YAML. The reconstruction pass that reads these boxes instead uses
``saved_bboxes.yaml``, as shown below.
Only ``detector_backend: rtdetr`` is accepted here -- ``saved_bboxes`` is the
other backend this fork supports, and it reads bbox JSONs rather than
producing them, so it has no role in a standalone detection run.

Usage (source-frame bboxes followed by perspective cropping):
    python inference/hand_detector_2d/scripts/run_hand_detector.py \\
        --input /path/to/video.mp4 \\
        --output_dir /path/to/bboxes/ \\
        --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml \\
        --intrinsics_json /path/to/intrinsics.json \\
        --no-rectify_pixel_aspect_ratio

    # Then run EgoHand on the same video with the same calibration:
    BBOX_DIR=/path/to/bboxes python inference/scripts/run_egohand_inference_pipeline.py \\
        --input /path/to/video.mp4 \\
        --bbox_detector_config inference/hand_detector_2d/configs/saved_bboxes.yaml \\
        --intrinsics_json /path/to/intrinsics.json \\
        --perspective_cropping \\
        --output_dir /path/to/results/

Keep the two stages consistent:

* Use the same video and calibration. For perspective cropping, pass
  ``--no-rectify_pixel_aspect_ratio`` in the detector stage to match live
  detection on source images. For standard pinhole cropping, match both
  stages' pixel-aspect settings to reproduce live detector preprocessing.
  Saved boxes are always mapped back to source coordinates. Omitting the
  flag can change detections when rectification resizes the images, but does
  not itself create a coordinate mismatch. FISHEYE624/KB4 calibrations
  require the flag because this tool's rectification helper rejects them.
* ``--max_seconds``. Frame IDs are the decode index ``{i:06d}`` from the start
  of the video, the same convention the pipeline uses. Truncation changes
  frame coverage without shifting IDs. Whole-clip tracking and smoothing can
  also change boxes on shared frames when the detector pass is truncated.

Frames are decoded in batches and only the detections are kept, so a long video
costs one batch of pixels rather than the whole clip. Tracking runs on the
accumulated detections after decoding finishes, so it sees the full sequence.

Dependencies: detection, the native tracker, the JSON writing, and
--intrinsics_json's pixel-aspect-ratio correction need this folder and the shared
``inference/media`` package plus numpy / opencv / pillow / pyyaml / onnxruntime,
plus huggingface_hub when downloading weights. Both folders can be copied with
their relative layout intact and run without the rest of EgoHand --
``pixel_aspect.py`` duplicates the square-pixel
resize math from ``inference/camera/square_pixel.py`` rather than importing
it, same reasoning as ``BBOX_FILE_PATTERN`` below.
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Running as a plain script (``python .../scripts/run_hand_detector.py``) leaves no package
    # context for the relative imports below to resolve against. Put the folder's
    # parent on sys.path and set the package context to ``rtdetr``. This also
    # works from a copy of the folder outside the repo and in the standalone
    # detector image.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "rtdetr"

from .detector import (  # noqa: E402
    RTDETR_HF_REPO,
    RTDetrConfig,
    boxes_from_detections,
    boxes_from_tracks,
    hf_download,
)
from inference.media.video_io import (  # noqa: E402
    find_videos,
    get_video_meta,
    read_video_frames_batched,
)

# Sibling of the rtdetr/ package, not inside it -- top-level import, not
# relative, since sys.path was pointed at the hand_detector_2d/ folder above (the
# same reason .detector resolves to rtdetr.detector rather than a relative
# ..rtdetr.detector).
from pixel_aspect import compute_square_pixel_resize_params, load_fx_fy  # noqa: E402

log = logging.getLogger("rtdetr_bbox")

SUPPORTED_BACKEND = "rtdetr"
DEFAULT_FRAME_BATCH = 256
# Duplicated from ``inference/hand_detector_2d/saved_bbox.py`` rather than
# imported: the standalone entrypoint uses the rtdetr package context and avoids
# importing the full inference package. SavedBBoxDetector derives ``stem`` from the
# containing folder's name -- see write_bbox_jsons.
BBOX_FILE_PATTERN = "{stem}_{hand}_bbox.json"


def load_rtdetr_config(path: str) -> tuple[RTDetrConfig, bool]:
    """Read a bbox detector YAML and return ``(RTDetrConfig, tracking)``.

    Mirrors ``run_egohand_inference_pipeline._load_detector_config``: same file
    format, same ``${ENV_VAR}`` expansion, same ``detector_backend`` /
    ``rtdetr_config`` / ``tracking`` keys. Every detector knob therefore lives
    in the YAML -- score threshold, padding, ONNX batch size, whether to track --
    and this tool grows no flags that could disagree with the pipeline's config
    for the same run.

    If ``rtdetr_config`` sets no ``weights``, falls back to downloading from HF
    (``RTDETR_HF_REPO``, filename from the ``rtdetr_hf_weights`` pseudo-field,
    default ``hand_detector.onnx``) -- same fallback
    ``inference/hand_detector_2d/configs/rtdetr.py`` uses for the pipeline's own
    rtdetr backend, so a shared config YAML behaves the same through either
    entrypoint.

    Raises:
        SystemExit: unreadable file, a non-rtdetr backend, or an unknown key or
            out-of-range value under ``rtdetr_config``. All of these are
            user-typed configuration, so they get a message, not a traceback.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "Reading a bbox detector config needs PyYAML (requirements.txt pins "
            "pyyaml==6.0.3). Install it with `pip install pyyaml`."
        ) from exc

    try:
        raw = Path(path).read_text()
    except OSError as exc:
        raise SystemExit(f"--bbox_detector_config {path}: {exc.strerror}") from exc
    cfg = yaml.safe_load(os.path.expandvars(raw)) or {}

    backend = cfg.get("detector_backend")
    if backend != SUPPORTED_BACKEND:
        raise SystemExit(
            f"{path}: detector_backend={backend!r}, but this tool only runs "
            f"{SUPPORTED_BACKEND!r} for now. Use "
            "inference/hand_detector_2d/configs/rtdetr.yaml or a copy of it. The other "
            "backends stay inside run_egohand_inference_pipeline.py, which wires up the "
            "per-video state they need."
        )

    rtdetr_config = dict(cfg.get("rtdetr_config") or {})
    rtdetr_hf_weights_path = rtdetr_config.pop(
        "rtdetr_hf_weights", "hand_detector.onnx"
    )
    try:
        config = RTDetrConfig.from_dict(rtdetr_config)
    except ValueError as exc:
        raise SystemExit(f"{path}: rtdetr_config: {exc}") from exc
    if not config.weights:
        weights = hf_download(rtdetr_hf_weights_path, repo_id=RTDETR_HF_REPO)
        log.info(
            "[rtdetr_bbox] weights not given, resolved via HF (%s/%s): %s",
            RTDETR_HF_REPO,
            rtdetr_hf_weights_path,
            weights,
        )
        config = dataclasses.replace(config, weights=weights)
    return config, bool(cfg.get("tracking", False))


def detect_video(detector, video_path, batch_size, max_frames, resize_params=None):
    """Decode the video in batches and return ``(image_ids, detections, (W, H))``.

    Frames go to the detector exactly as decoded, unless ``resize_params`` is
    given -- then each frame is rectified to square pixels first. This matches
    live detection with standard pinhole cropping when both passes use the same
    pixel-aspect setting. Perspective cropping skips this full-frame resize;
    leave ``resize_params`` unset to match live detection in that mode.

    Only detections survive a batch, so peak memory is one batch of frames
    regardless of clip length. When nothing decodes, the IDs and detections come
    back empty and the size is ``None``; the caller skips the video on that.
    """
    image_ids: list[str] = []
    detections: list = []
    size: tuple[int, int] | None = None
    for batch in read_video_frames_batched(str(video_path), batch_size, max_frames):
        if resize_params is not None:
            # Replace each frame in place so peak memory does not double --
            # matches run_egohand_inference_pipeline.py's own rectification loop.
            for i in range(len(batch)):
                idx, frame = batch[i]
                batch[i] = (idx, resize_params.apply_to_frame(frame))
        frames = [frame for _, frame in batch]
        if size is None:
            height, width = frames[0].shape[:2]
            size = (width, height)
        detections.extend(detector.detect_frames(frames))
        image_ids.extend(f"{idx:06d}" for idx, _ in batch)
        log.info("[rtdetr_bbox] detected %d frame(s) so far", len(image_ids))
    return image_ids, detections, size


def boxes_for_video(detector, image_ids, detections, hands, size, *, fps):
    """Convert detections to boxes using the detector's tracking settings."""
    width, _ = size
    if detector.use_tracker:
        bbox_per_hand, info = boxes_from_tracks(
            image_ids,
            detections,
            hands,
            image_width=width,
            enforce_lr_order=detector.enforce_lr_order,
            drop_gap_filled=detector.drop_gap_filled,
            smoothing=detector.smoothing,
            frame_times=[int(image_id) / fps for image_id in image_ids],
        )
        detector.log_tracking(info, hands)
        return bbox_per_hand

    return boxes_from_detections(image_ids, detections, hands)


def write_bbox_jsons(bbox_per_hand, out_dir: Path, hands) -> None:
    """Write one JSON per hand in the layout ``SavedBBoxDetector`` expects.

    That reader derives the filename stem from the containing folder's name, so
    the folder has to be named after the video -- hence ``out_dir.name`` here
    rather than a stem threaded down from the caller.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for hand in hands:
        boxes = bbox_per_hand.get(hand, {})
        path = out_dir / BBOX_FILE_PATTERN.format(stem=out_dir.name, hand=hand)
        path.write_text(
            json.dumps(
                {
                    image_id: [float(v) for v in bbox]
                    for image_id, bbox in boxes.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("[rtdetr_bbox] wrote %s (%d frame(s))", path, len(boxes))


def rewrite_bbox_json_in_original_frame(out_dir: Path, hands, resize_params) -> None:
    """Rewrite bbox JSON just written from the corrected frame to the source frame.

    Detection ran on the corrected (rectified) frame when ``resize_params``
    was given, so ``write_bbox_jsons`` above persisted boxes in that frame --
    mapping once here, on the way out, matches what
    ``run_egohand_inference_pipeline.py`` does for its own live rtdetr
    detections, so ``SavedBBoxDetector``'s ``reports_in_source_frame``
    contract holds no matter which tool produced the JSON. Same math as
    ``inference.camera.square_pixel.rewrite_bbox_json_in_original_frame``,
    duplicated for the reasons in the module docstring.
    """
    if resize_params is None or resize_params.is_identity:
        return
    for hand in hands:
        path = out_dir / BBOX_FILE_PATTERN.format(stem=out_dir.name, hand=hand)
        if not path.exists():
            continue
        boxes = json.loads(path.read_text())
        path.write_text(
            json.dumps(
                {
                    frame_id: resize_params.to_original_bbox(bbx).tolist()
                    for frame_id, bbx in boxes.items()
                }
            )
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="RT-DETRv2 hand bbox detection on videos, saved as per-hand JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, help="MP4 file or folder of MP4 files.")
    p.add_argument(
        "--output_dir",
        required=True,
        help="Root for the bbox JSONs; one <video_stem>/ subfolder per video. "
        "Pass this same path as BBOX_DIR to the saved_bboxes.yaml second pass.",
    )
    p.add_argument(
        "--bbox_detector_config",
        required=True,
        help="Bbox detector YAML, the same format run_egohand_inference_pipeline.py takes. "
        "Must set detector_backend: rtdetr. Everything about the detector comes from "
        "this file -- weights, score threshold, padding, ONNX batch size, whether to "
        "track. See inference/hand_detector_2d/configs/rtdetr.yaml.",
    )
    p.add_argument("--hand", choices=["left", "right", "both"], default="both")
    p.add_argument(
        "--device", default="cuda", help="Torch-style device, e.g. cuda:1 or cpu."
    )
    p.add_argument(
        "--intrinsics_json",
        default=None,
        help="Camera calibration JSON, same format as the pipeline's --intrinsics_json. "
        "Only needed to decide the --rectify_pixel_aspect_ratio correction; omit for "
        "square-pixel cameras or when this detector pass uses "
        "--no-rectify_pixel_aspect_ratio.",
    )
    p.add_argument(
        "--rectify_pixel_aspect_ratio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize non-square-pixel images before detection, then map boxes back "
        "to source coordinates before writing JSON. Default: on. Use "
        "--no-rectify_pixel_aspect_ratio when the second pass uses "
        "--perspective_cropping. For standard pinhole cropping, match both passes' "
        "pixel-aspect settings to reproduce live detector preprocessing. Resizing "
        "can change detections, but saved boxes remain in source coordinates. "
        "No effect without --intrinsics_json or when the focal-length ratio is "
        "within the resize tolerance of 1.",
    )
    p.add_argument(
        "--max_seconds",
        type=float,
        default=None,
        help="Only process the first N seconds, matching the pipeline's --max_seconds.",
    )
    p.add_argument(
        "--frame_batch_size",
        type=int,
        default=DEFAULT_FRAME_BATCH,
        help=f"Frames decoded into RAM at once (default: {DEFAULT_FRAME_BATCH}). Caps peak "
        "memory on long clips; tracking still sees the whole sequence. Distinct from the "
        "ONNX forward-pass batch, which the YAML sets via the rtdetr_config batch_size: key.",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    videos = find_videos(Path(args.input))
    if not videos:
        raise SystemExit(f"No MP4 files found at {args.input}")

    config, tracking = load_rtdetr_config(args.bbox_detector_config)
    # Use the same detector construction as the full EgoHand pipeline.
    detector = config.build(device=args.device, tracking=tracking)

    output_dir = Path(args.output_dir)
    log.info(
        "[rtdetr_bbox] %d video(s), hands=%s, tracking=%s (config: %s) -> %s",
        len(videos),
        ",".join(hands),
        tracking,
        args.bbox_detector_config,
        output_dir,
    )

    for video_path in videos:
        meta = get_video_meta(str(video_path))
        max_frames = (
            int(args.max_seconds * meta["fps"])
            if args.max_seconds is not None
            else None
        )

        log.info(
            "[rtdetr_bbox] === %s === (%d frames, %dx%d @ %.1f fps)",
            video_path.name,
            meta["num_frames"],
            meta["width"],
            meta["height"],
            meta["fps"],
        )

        # Mirrors the pipeline's rectification for standard pinhole cropping.
        # For a perspective-cropping second pass, disable rectification here
        # to match live detection on source images. resize_params stays None
        # when disabled, when no calibration is given, or for square pixels.
        resize_params = None
        if args.intrinsics_json and args.rectify_pixel_aspect_ratio:
            fx, fy = load_fx_fy(args.intrinsics_json, meta["width"], meta["height"])
            params = compute_square_pixel_resize_params(
                fx, fy, meta["width"], meta["height"]
            )
            if not params.is_identity:
                resize_params = params
                log.info(
                    "[rtdetr_bbox] pixel-aspect rectify %dx%d -> %dx%d "
                    "(sx=%.4f, sy=%.4f, calibration %s); bbox JSON written "
                    "in the original frame",
                    meta["width"],
                    meta["height"],
                    params.width,
                    params.height,
                    params.scale_x,
                    params.scale_y,
                    args.intrinsics_json,
                )

        image_ids, detections, size = detect_video(
            detector,
            video_path,
            max(1, args.frame_batch_size),
            max_frames,
            resize_params=resize_params,
        )
        if not image_ids:
            log.warning(
                "[rtdetr_bbox] SKIP: no frames decoded from %s", video_path.name
            )
            continue

        bbox_per_hand = boxes_for_video(
            detector, image_ids, detections, hands, size, fps=meta["fps"]
        )
        video_out_dir = output_dir / video_path.stem
        write_bbox_jsons(bbox_per_hand, video_out_dir, hands)
        rewrite_bbox_json_in_original_frame(video_out_dir, hands, resize_params)

    log.info("[rtdetr_bbox] done -> %s", output_dir)


if __name__ == "__main__":
    main()

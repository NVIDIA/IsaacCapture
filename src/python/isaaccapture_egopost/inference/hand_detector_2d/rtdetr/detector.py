# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Detect 2D hand bounding boxes with an RT-DETRv2 ONNX detector.

Implements the ``HandBBoxDetector`` Protocol with
``{image_id: {hand: [cx, cy, size]}}`` output and optional per-hand JSON files.
The model detects hand boxes directly using ``onnxruntime``. ONNX Runtime and
OpenCV are imported lazily so the box-processing helpers can run without them.

Two temporal modes:

* ``use_tracker=False`` (default): per-frame detection. Boxes are top-1 per hand,
  padded, squared, and not clipped. Frames with no confident detection are
  omitted from the result.
* ``use_tracker=True``: the RT-DETR-native whole-sequence tracker in
  ``tracking``. It resolves left/right identity temporally, rescues
  sub-threshold detections, and fills gaps for hands detected in the clip.
  An off-screen hand can therefore still get an interpolated box. Pass
  ``drop_gap_filled=True`` to emit only frames with a real detection.

Padding convention: ``padding`` here follows the EgoHand convention where
the final square side is ``max(w, h) * (1 + padding)``. The vendored
``refine_box`` uses a per-side fraction instead (side ``* (1 + 2 * pad)``), so
the constructor halves it. ``use_tracker=True`` ignores ``padding`` entirely --
the tracker applies its own ``PAD``/``MIN_SIZE_FRAC`` and requires an unpadded
detector, which is exactly the double-padding trap this note exists to flag.
"""

import dataclasses
import json
import logging
from math import log1p
from pathlib import Path

import numpy as np

from .bbox_smoothing import (
    REFERENCE_FPS,
    BBoxSmoothingConfig,
    smooth_bbox_track,
    validate_frame_times,
)
from .tracking import (
    MIN_SIZE_FRAC,
    PAD,
    THR_RESCUE,
    HandTracker,
)

log = logging.getLogger(__name__)

DEFAULT_SCORE_THR = 0.5
DEFAULT_BATCH_SIZE = 8
# Reject boxes whose side is at most one pixel.
MIN_VALID_BOX_SIZE = 1.0

RTDETR_HF_REPO = "nvidia/RT-DETR-Hand-Detector-v1"


def hf_download(filename: str, repo_id: str = RTDETR_HF_REPO) -> str:
    """Download ``filename`` from an HF repo (default: RTDETR_HF_REPO), caching locally.

    Defined here, not imported from ``inference.assets`` (which has
    an equivalent), so this whole ``hand_detector_2d`` folder stays self-contained
    -- it's baked into the standalone detector Docker image on its own, without
    the rest of ``inference/`` (see ``inference/docker/Dockerfile.detector`` and
    this package's own module docstring). ``huggingface_hub`` is imported lazily
    for the same reason every other heavy dependency in this module is: importing
    this module, or anything that imports it, must not require it upfront.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def _providers_for_device(device: str) -> list:
    """Map a torch-style device string onto onnxruntime execution providers.

    ``cuda:3`` has to become ``("CUDAExecutionProvider", {"device_id": 3})`` --
    ORT does not observe ``torch.cuda.set_device``, so a caller that shards work
    across GPUs by device string would otherwise pile every worker onto device 0.
    """
    device = str(device)
    if not device.startswith("cuda"):
        return ["CPUExecutionProvider"]
    device_id = int(device.split(":", 1)[1]) if ":" in device else 0
    return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]


def _is_valid_box(bbox) -> bool:
    return bool(np.isfinite(bbox).all() and bbox[2] > MIN_VALID_BOX_SIZE)


def boxes_from_detections(image_ids, detections, hands) -> dict:
    """Per-frame boxes: the highest-scoring detection per hand, or nothing.

    Args:
        image_ids: Image IDs aligned with ``detections``.
        detections: Per-frame lists of ``onnx_model.Detection``, score-sorted.
        hands: Requested hand sides.

    Returns:
        ``{hand: {image_id: np.float32([cx, cy, size])}}``, sparse -- a frame
        with no detection for a hand is simply absent from that hand's dict.
    """
    from .onnx_model import LABEL_ID_TO_HAND

    bbox_per_hand: dict[str, dict[str, np.ndarray]] = {hand: {} for hand in hands}
    for image_id, dets in zip(image_ids, detections):
        for det in dets:
            hand = LABEL_ID_TO_HAND.get(det.label_id)
            if hand not in bbox_per_hand or image_id in bbox_per_hand[hand]:
                continue  # unknown class id, unrequested hand, or top-1 already taken
            bbox = np.asarray(det.cxcys, dtype=np.float32)
            if _is_valid_box(bbox):
                bbox_per_hand[hand][image_id] = bbox
    return bbox_per_hand


def boxes_from_tracks(
    image_ids,
    detections,
    hands,
    image_width: int,
    enforce_lr_order: bool = True,
    drop_gap_filled: bool = False,
    *,
    smoothing: BBoxSmoothingConfig | None = None,
    frame_times=None,
) -> tuple[dict, dict]:
    """Whole-sequence tracked boxes.

    Args:
        image_ids: Image IDs aligned with ``detections``, in temporal order.
        detections: Per-frame lists of ``onnx_model.Detection``.
        hands: Requested hand sides.
        image_width: Full-image width; sets the tracker's minimum box size.
        enforce_lr_order: Force smaller-x to be the left hand on both-present frames.
        drop_gap_filled: Emit only frames that carried a real detection.
        smoothing: Optional offline post-filter, applied before sparse emission.
        frame_times: Timestamps in seconds aligned with image_ids; required
            when smoothing is enabled. Supply the complete clip in one call.

    Returns:
        ``({hand: {image_id: np.float32([cx, cy, size])}}, info)``. ``info`` is
        report-only and drives the caller's logging: ``per_hand[hand]`` carries
        ``frames`` / ``detected`` / ``emitted`` / ``gap_filled`` /
        ``never_detected``, and ``rescued`` / ``label_overridden`` / ``crossed``
        are clip totals across both hands (the tracker counts them globally).

    This function does no logging -- ``RTDetrHandBBoxDetector.run`` reports the
    whole clip at once, so the numbers appear before the conclusions drawn
    from them.
    """
    if smoothing is not None:
        frame_times = validate_frame_times(frame_times, len(image_ids))
    tracker = HandTracker(image_width=image_width, enforce_lr_order=enforce_lr_order)
    tracks = tracker.track(detections)
    raw_present = tracks["raw_present"]
    if smoothing is not None:
        # Process both tracks so requesting one hand yields the same boxes as
        # requesting both, including any left/right order correction below.
        smoothed_tracks = {
            hand: smooth_bbox_track(
                tracks[hand], frame_times, raw_present[hand], smoothing
            )
            if raw_present[hand].any()
            else tracks[hand]
            for hand in ("left", "right")
        }
        if enforce_lr_order and all(
            raw_present[hand].any() for hand in ("left", "right")
        ):
            left, right = smoothed_tracks["left"], smoothed_tracks["right"]
            native_left, native_right = tracks["left"][:, 0], tracks["right"][:, 0]
            crossed = (native_left <= native_right) & (left[:, 0] > right[:, 0])
            # Undo only crossings introduced by independent smoothing. This
            # boundary moves each x-center toward its own native center, so
            # coverage remains valid without changing sizes or hand identities.
            boundary = np.clip(
                (left[crossed, 0] + right[crossed, 0]) / 2,
                native_left[crossed],
                native_right[crossed],
            )
            left[crossed, 0] = right[crossed, 0] = boundary
        tracks.update(smoothed_tracks)

    bbox_per_hand: dict[str, dict[str, np.ndarray]] = {}
    per_hand_info: dict[str, dict] = {}
    for hand in hands:
        boxes, present = tracks[hand], raw_present[hand]
        # apply_bbox_rules short-circuits when a hand was never detected and
        # hands back all-zero boxes flagged present. Those would reach
        # crop_and_preprocess as size=0 and divide by zero, so emit nothing.
        emitted = (
            {}
            if not present.any()
            else {
                image_id: boxes[index].astype(np.float32)
                for index, image_id in enumerate(image_ids)
                if (present[index] or not drop_gap_filled)
                and _is_valid_box(boxes[index])
            }
        )
        detected = int(present.sum())
        bbox_per_hand[hand] = emitted
        per_hand_info[hand] = {
            "frames": len(image_ids),
            "detected": detected,
            "emitted": len(emitted),
            "gap_filled": max(0, len(emitted) - detected),
            "never_detected": detected == 0,
        }
    stats = tracks["stats"]
    info = {
        "per_hand": per_hand_info,
        "rescued": stats["rescued"],
        "label_overridden": stats["label_ignored"],
        "crossed": stats["crossed"],
    }
    return bbox_per_hand, info


class RTDetrHandBBoxDetector:
    """Detect 2D hand bounding boxes with RT-DETRv2 (ONNX).

    The ONNX session is created once at construction time and reused across all
    ``run()`` calls. Satisfies the ``HandBBoxDetector`` Protocol.

    Args:
        weights: Path to the RT-DETRv2 ``.onnx`` file. Required -- callers pass
            their own rather than inheriting a default here.
        device: Torch-style device string; mapped onto ORT providers.
        batch_size: Frames per ONNX forward pass. Small on purpose: the model is
            launch-bound, so large batches buy nothing but memory.
        padding: Fractional bbox padding, EgoHand convention (side ``* (1 + padding)``).
            Ignored when ``use_tracker`` is true.
        score_thr: Minimum detection confidence. Ignored when ``use_tracker`` is
            true -- the tracker needs sub-threshold candidates for its rescue rule
            and runs the detector at ``THR_RESCUE``.
        use_tracker: Use the RT-DETR-native whole-sequence tracker.
        min_size: Minimum square side in pixels for the per-frame path.
        clip: Clip per-frame detections to image bounds before conversion to
            square crops. Default false. Ignored with tracking: detector
            candidates are always clipped, but final tracks can extend outside
            the image. EgoHand zero-pads out-of-frame crops.
        enforce_lr_order: Tracker only. When true (the upstream default), the
            smaller-x box is forced to be the left hand on frames where both
            hands are present.
        drop_gap_filled: Tracker only. Emit boxes only for frames that carried a
            real detection, restoring the sparse VitPose-comparable behaviour.
        providers: Explicit onnxruntime providers, overriding ``device``.
        smoothing: Optional native-track post-filter. Requires complete clips
            and frame timestamps in run().
    """

    # run() writes to output_paths_per_hand when given one.
    writes_bbox_json = True

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        batch_size: int = DEFAULT_BATCH_SIZE,
        padding: float = 0.3,
        score_thr: float = DEFAULT_SCORE_THR,
        use_tracker: bool = False,
        min_size: float = 0.0,
        clip: bool = False,
        enforce_lr_order: bool = True,
        drop_gap_filled: bool = False,
        providers: list | None = None,
        smoothing: BBoxSmoothingConfig | None = None,
    ):
        if smoothing is not None and not use_tracker:
            raise ValueError("BBox smoothing requires use_tracker=True")
        if not weights:
            raise ValueError(
                "weights must be a non-empty path to the RT-DETRv2 .onnx file. "
                "Only construct RTDetrHandBBoxDetector when bbox detection is enabled."
            )
        # Imported here, not at module scope: the CLI entrypoints import this
        # module unconditionally, and they must keep working with saved-bbox
        # or ground-truth detection in an env without onnxruntime.
        from .onnx_model import HandDetector

        self.batch_size = max(1, int(batch_size))
        self.padding = padding
        self.use_tracker = use_tracker
        self.enforce_lr_order = enforce_lr_order
        self.drop_gap_filled = drop_gap_filled
        self.smoothing = smoothing

        if use_tracker:
            # The tracker owns padding and thresholding. Handing it a detector
            # with post-processing on would double-pad every box.
            detector_kwargs = dict(
                score_thr=THR_RESCUE,
                top1_per_class=False,
                pad=0.0,
                square=False,
                min_size=0.0,
                clip=True,
            )
        else:
            detector_kwargs = dict(
                score_thr=score_thr,
                top1_per_class=True,
                pad=padding
                / 2.0,  # EgoHand convention (1 + padding) -> per-side (1 + 2 * pad)
                square=True,
                min_size=min_size,
                clip=clip,
            )
        log.info(
            "[RTDetr] weights=%s device=%s mode=%s", weights, device, self.mode_name
        )
        # Spell out the effective geometry rather than dumping kwargs. Both modes
        # produce numbers that differ from the flags the caller passed -- the
        # per-frame path halves `padding` for refine_box's per-side convention,
        # and tracker mode overrides `score_thr` and `padding` outright -- so a
        # log that only showed the final values would read like the flags were
        # being ignored.
        if use_tracker:
            log.info(
                "[RTDetr] the tracker owns box geometry, so the detector runs raw: "
                "score_thr %.2f -> %.2f (keeps sub-threshold candidates for the rescue "
                "rule), padding %.2f -> %.2f (tracker PAD=%.2f), min_size -> "
                "%.2f x image_width",
                score_thr,
                THR_RESCUE,
                padding,
                PAD - 1.0,
                PAD,
                MIN_SIZE_FRAC,
            )
        else:
            log.info(
                "[RTDetr] boxes: top-1 per hand, score_thr=%.2f, padding=%.2f "
                "(square side x %.2f, i.e. refine_box pad=%.3f per side), clip=%s, "
                "min_size=%.0f px",
                score_thr,
                padding,
                1.0 + padding,
                padding / 2.0,
                clip,
                min_size,
            )
        self.detector = HandDetector(
            weights=weights,
            providers=providers
            if providers is not None
            else _providers_for_device(device),
            **detector_kwargs,
        )
        log.info("[RTDetr] onnxruntime providers: %s", self.detector.providers)

    @property
    def requires_full_clip(self) -> bool:
        """Offline bbox smoothing must run before cropping a complete clip."""
        return self.smoothing is not None

    @property
    def mode_name(self) -> str:
        """Human-readable temporal mode, used in every log line about this run."""
        if not self.use_tracker:
            return "per-frame"
        mode = "native-tracker+sparse" if self.drop_gap_filled else "native-tracker"
        return mode + "+bbox-smoothing" if self.smoothing is not None else mode

    def detect_frames(self, images_bgr) -> list:
        """Raw per-frame detections, no aggregation and no left/right assignment.

        ``run`` is the whole-clip entry point and needs every frame in RAM at
        once. This one exists for callers that decode a long video in chunks:
        detections are tiny next to frames, so accumulating them chunk by chunk
        and aggregating once at the end -- via ``boxes_from_detections`` or
        ``boxes_from_tracks`` -- is equivalent to a single ``run`` call while
        holding only one chunk of pixels at a time.
        """
        return self._detect_per_frame(images_bgr)

    def _detect_per_frame(self, images_bgr) -> list:
        """Run the ONNX detector over every frame, in ``batch_size`` chunks."""
        import cv2
        from PIL import Image

        detections = []
        for start in range(0, len(images_bgr), self.batch_size):
            chunk = images_bgr[start : start + self.batch_size]
            pil_chunk = [
                Image.fromarray(
                    cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
                )
                for image in chunk
            ]
            detections.extend(self.detector.detect_batch(pil_chunk))
        return detections

    def run(
        self,
        image_ids,
        images_bgr,
        hands,
        output_paths_per_hand=None,
        *,
        frame_times: dict[str, float] | None = None,
    ) -> dict:
        """Detect per-image hand bboxes.

        Args:
            image_ids: Image IDs aligned with ``images_bgr``. Assumed to be in
                temporal order when ``use_tracker`` is true.
            images_bgr: Preloaded images in OpenCV BGR channel order.
            hands: Hand sides to detect, usually ``["left", "right"]``.
            output_paths_per_hand: Optional destination bbox JSON path keyed by
                hand side. When ``None``, no JSON files are written.
            frame_times: Image ID to timestamp in seconds. Required for bbox
                smoothing; each run must contain a complete clip.

        Returns:
            ``{image_id: {hand: np.array([cx, cy, size])}}``. Image IDs with no
            bbox for a given hand are absent from that hand's entry.
        """
        image_ids = [str(image_id) for image_id in image_ids]
        times = None
        if self.smoothing is not None:
            if frame_times is None or any(i not in frame_times for i in image_ids):
                raise ValueError(
                    "BBox smoothing requires frame_times for every image ID"
                )
            times = validate_frame_times(
                [frame_times[i] for i in image_ids], len(image_ids)
            )
        if not image_ids:
            return {}
        log.info(
            "[RTDetr] %d image(s), hands=%s, mode=%s",
            len(images_bgr),
            ",".join(hands),
            self.mode_name,
        )
        detections = self._detect_per_frame(images_bgr)

        if self.use_tracker:
            widths = {image.shape[1] for image in images_bgr}
            if len(widths) > 1:
                log.warning(
                    "[RTDetr] mixed frame widths %s; the tracker's minimum box size "
                    "uses the first (%d px)",
                    sorted(widths),
                    images_bgr[0].shape[1],
                )
            bbox_per_hand, info = boxes_from_tracks(
                image_ids,
                detections,
                hands,
                image_width=images_bgr[0].shape[1],
                enforce_lr_order=self.enforce_lr_order,
                drop_gap_filled=self.drop_gap_filled,
                smoothing=self.smoothing,
                frame_times=times,
            )
            self.log_tracking(info, hands)
        else:
            bbox_per_hand = boxes_from_detections(image_ids, detections, hands)

        bboxes_per_image: dict[str, dict[str, np.ndarray]] = {
            image_id: {} for image_id in image_ids
        }
        for hand in hands:
            bbox_dict = bbox_per_hand[hand]
            if output_paths_per_hand is not None:
                output_path = Path(output_paths_per_hand[hand])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Merge into any existing JSON rather than overwriting it: run() is
                # called once per frame batch with the same output path, so batches
                # must accumulate instead of the last one clobbering earlier ones.
                merged = (
                    json.loads(output_path.read_text()) if output_path.exists() else {}
                )
                merged.update({k: [float(v) for v in b] for k, b in bbox_dict.items()})
                with open(output_path, "w") as f:
                    json.dump(merged, f, indent=2)
                log.info("[Saved] %s  (%d image(s))", output_path, len(bbox_dict))
            num_without = len(image_ids) - len(bbox_dict)
            # Tracker mode already reported per-hand coverage, with the reason;
            # repeating it here would contradict it ("no reliable bbox" is not
            # what happened when a whole track was dropped as degenerate).
            if num_without and not self.use_tracker:
                log.info(
                    "[Skip] %d image(s) had no reliable %s hand bbox", num_without, hand
                )
            for image_id, bbox in bbox_dict.items():
                bboxes_per_image[image_id][hand] = np.asarray(bbox, dtype=np.float32)
        return bboxes_per_image

    @staticmethod
    def log_tracking(info, hands) -> None:
        """Report the whole tracked clip: coverage per hand, then clip totals.

        Written for someone reading only the log: every number says what it is
        counting and out of what, and the per-hand line states why a hand ended
        up with fewer boxes than frames.
        """
        for hand in hands:
            hand_info = info["per_hand"][hand]
            gaps = hand_info["frames"] - hand_info["detected"]
            if hand_info["never_detected"]:
                reason = "hand never detected in this clip, so no track was emitted"
            elif hand_info["gap_filled"]:
                reason = f"{hand_info['gap_filled']} gap-filled"
            elif gaps:
                reason = f"{gaps} gap frame(s) dropped by drop_gap_filled"
            else:
                reason = "detected on every frame"
            log.info(
                "[RTDetrTracking] %s: detected on %d/%d frame(s) -> %d box(es) emitted (%s)",
                hand,
                hand_info["detected"],
                hand_info["frames"],
                hand_info["emitted"],
                reason,
            )
        # These three are clip totals over both hands, not per-hand: the tracker
        # counts them once per frame decision.
        log.info(
            "[RTDetrTracking] clip totals: %d sub-threshold detection(s) rescued, "
            "%d frame(s) where track continuity overrode the class label, "
            "%d frame(s) with the hands crossed in x",
            info["rescued"],
            info["label_overridden"],
            info["crossed"],
        )


@dataclasses.dataclass(frozen=True)
class RTDetrConfig:
    """RT-DETR settings shared by standalone detection and EgoHand inference.

    CLI entry points read the YAML ``rtdetr_config`` mapping via ``from_dict``.
    Library callers can also use ``from_spec`` with comma-separated values,
    for example ``weights=hand.onnx,batch_size=12,score_thr=0.6``. The YAML's
    top-level ``tracking`` flag is passed separately to ``build``.

    Attributes:
        weights: Path to the RT-DETRv2 ``.onnx``. YAML loaders download the default
            HF weights when this is None or empty. Direct build() callers must
            supply a non-empty path.
        batch_size: Frames per ONNX forward pass. Small on purpose -- the model
            is launch-bound, so bigger batches only cost memory.
        score_thr: Minimum detection confidence. Ignored under the native tracker,
            which needs sub-threshold candidates for its rescue rule.
        padding: Fractional bbox padding, EgoHand convention (side ``* (1 + padding)``).
            Ignored under the native tracker, which applies its own.
        clip: Clip per-frame detections before square-crop conversion. Off by
            default; ignored with tracking, which clips detector candidates.
        lr_order: Native tracker only. Force the smaller-x box to be the left
            hand on frames where both are present.
        drop_gap_filled: Native tracker only. Emit boxes only on frames that
            carried a real detection.
        bbox_smoothing: Enable offline tighter crops and temporal smoothing.
            Requires tracking=True.
        bbox_size_scale: Target side as a fraction of the native side. The crop
            guard and rate limit can enlarge it.
        bbox_size_alpha_grow: Growth EMA weight at 20 fps, in (0, 1].
        bbox_size_alpha_shrink: Shrink EMA weight at 20 fps, in (0, 1].
        bbox_max_size_change: Maximum fractional growth per 20-fps interval.
            Contraction uses the reciprocal ratio; resets are exempt.
    """

    weights: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    score_thr: float = DEFAULT_SCORE_THR
    padding: float = 0.3
    clip: bool = False
    lr_order: bool = True
    drop_gap_filled: bool = False
    bbox_smoothing: bool = False
    bbox_size_scale: float = 0.8
    bbox_size_alpha_grow: float = 0.12
    bbox_size_alpha_shrink: float = 0.06
    bbox_max_size_change: float = 0.01

    @classmethod
    def from_spec(cls, spec: str) -> "RTDetrConfig":
        """Parse ``"key=value,key=value"``. Empty string yields all defaults.

        Raises:
            ValueError: unknown key, malformed item, or invalid value.
        """
        fields = {f.name: f for f in dataclasses.fields(cls)}
        values: dict = {}
        for item in (chunk.strip() for chunk in spec.split(",")):
            if not item:
                continue  # tolerate trailing/doubled commas
            key, sep, raw = item.partition("=")
            key = key.strip()
            if not sep:
                raise ValueError(
                    f"--rtdetr item {item!r} is not key=value; "
                    f"valid keys: {', '.join(sorted(fields))}"
                )
            if key not in fields:
                raise ValueError(
                    f"--rtdetr has no key {key!r}; valid keys: {', '.join(sorted(fields))}"
                )
            values[key] = cls._coerce(key, raw.strip(), fields[key].type)

        config = cls(**values)
        cls._validate(config, prefix="--rtdetr")
        return config

    @classmethod
    def from_dict(cls, data: dict | None) -> "RTDetrConfig":
        """Build directly from a parsed YAML mapping (the ``rtdetr_config:`` block).

        Values are already typed by the YAML parser (int/float/bool/str), so
        unlike :meth:`from_spec` no string coercion is needed -- this is a plain
        keyword expansion. ``None``/empty yields all defaults.

        Raises:
            ValueError: unknown key or invalid value.
        """
        fields = {f.name for f in dataclasses.fields(cls)}
        data = data or {}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(
                f"rtdetr_config has unknown key(s) {sorted(unknown)}; "
                f"valid keys: {', '.join(sorted(fields))}"
            )
        config = cls(**data)
        cls._validate(config, prefix="rtdetr_config")
        return config

    @classmethod
    def _validate(cls, config: "RTDetrConfig", *, prefix: str):
        """Shared range checks for both from_spec and from_dict."""
        if not isinstance(config.bbox_smoothing, bool):
            raise ValueError(f"{prefix} bbox_smoothing must be a boolean")
        config.smoothing_config()
        if config.batch_size < 1:
            raise ValueError(f"{prefix} batch_size={config.batch_size} must be >= 1")

    @staticmethod
    def _coerce(key: str, raw: str, annotation):
        """Convert one spec value to the field's declared type."""
        # Annotations are strings under `from __future__ import annotations`, and
        # plain types otherwise; matching on the text covers both.
        text = (
            annotation
            if isinstance(annotation, str)
            else getattr(annotation, "__name__", str(annotation))
        )
        if "bool" in text:
            lowered = raw.lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            raise ValueError(
                f"--rtdetr {key}={raw!r} is not a boolean (use true/false)"
            )
        if "int" in text:
            try:
                return int(raw)
            except ValueError:
                raise ValueError(f"--rtdetr {key}={raw!r} is not an integer") from None
        if "float" in text:
            try:
                return float(raw)
            except ValueError:
                raise ValueError(f"--rtdetr {key}={raw!r} is not a number") from None
        return raw

    def smoothing_config(self) -> BBoxSmoothingConfig:
        """Validate user-facing controls and construct the native post-filter."""
        # Validate the fractional rate before log1p so malformed YAML gets a
        # useful field name rather than a math-domain/type error.
        value = self.bbox_max_size_change
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "rtdetr_config bbox_max_size_change must be finite and nonnegative"
            )
        try:
            return BBoxSmoothingConfig(
                target_size_scale=self.bbox_size_scale,
                size_alpha_grow=self.bbox_size_alpha_grow,
                size_alpha_shrink=self.bbox_size_alpha_shrink,
                max_size_log_rate_per_second=REFERENCE_FPS * log1p(value),
            )
        except ValueError as exc:
            raise ValueError(f"rtdetr_config bbox smoothing: {exc}") from exc

    def build(self, *, device: str, tracking: bool = False):
        """Construct the detector this config describes.

        Args:
            device: Torch-style device string, e.g. ``"cuda"`` or ``"cuda:3"``.
            tracking: Enable the native whole-sequence tracker. False uses
                independent per-frame detections. Bbox smoothing requires True.

        Returns:
            A ``HandBBoxDetector``.
        """
        self._validate(self, prefix="rtdetr_config")
        if self.bbox_smoothing and not tracking:
            raise ValueError("bbox_smoothing requires tracking=true")
        return RTDetrHandBBoxDetector(
            weights=self.weights,
            device=device,
            batch_size=self.batch_size,
            padding=self.padding,
            score_thr=self.score_thr,
            use_tracker=tracking,
            clip=self.clip,
            enforce_lr_order=self.lr_order,
            drop_gap_filled=self.drop_gap_filled,
            smoothing=self.smoothing_config() if self.bbox_smoothing else None,
        )

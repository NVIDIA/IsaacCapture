# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import json
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable


class PerfStage(Enum):
    MODEL_LOAD = "model_load"  # one-time: checkpoint load + model construction + device placement
    BBOX_DETECTION = "bbox_detection"  # per-frame: pipeline.bbox_detector.run(...), whichever backend is configured
    INFERENCE = "inference"  # per-frame: the whole predictor.run() call -- superset of the five stages below (adaptive_smoothing only when not deferred)
    PREPROCESSING = "preprocessing"  # per-frame: hand-crop preparation (perspective cropping OR bbox crop+resize, mutually exclusive per run)
    BATCH_ASSEMBLY = "batch_assembly"  # per-frame: stacking crops/cameras into batch tensors + host-to-device copy, before forward
    MODEL_FORWARD = "model_forward"  # per-frame: model.forward_step(...) only, CUDA-synchronized for an accurate GPU-inclusive number
    POST_FORWARD = "post_forward"  # per-frame: camera math, right-hand mirror, device-to-host copy, result-dict assembly
    ADAPTIVE_SMOOTHING = "adaptive_smoothing"  # per-frame: smooth_adaptive_hand_results(...), only recorded when --postprocessing_smooth is on
    OVERLAY_RENDERING = "overlay_rendering"  # per-frame: render_clip(...) -- per-frame drawing plus video encoding, skipped with --no_save_videos
    TOTAL_WALL = "total_wall"  # per-frame: the whole per-video processing block (bbox detection through overlay rendering), for sanity-checking that the other stages sum close to this


#: Human-readable description per stage, included in LocalPerfWriter's JSON
#: output so perf.json is self-explanatory without cross-referencing this file
#: or inference/perf/README.md.
STAGE_DESCRIPTIONS: dict[PerfStage, str] = {
    PerfStage.MODEL_LOAD: (
        "One-time cost: checkpoint load, model construction, and device "
        "placement. Recorded once before the per-video loop, so it is "
        "amortized across every video in a multi-video run."
    ),
    PerfStage.BBOX_DETECTION: (
        "Per-frame wall time of hand bbox detection, whichever backend is "
        "configured (e.g. rtdetr, or the JSON-read time for saved_bboxes)."
    ),
    PerfStage.INFERENCE: (
        "Per-frame wall time of the whole EgoHand predictor call. A superset "
        "of preprocessing, batch_assembly, model_forward, post_forward, and "
        "adaptive_smoothing, which run one after another inside it and do "
        "not overlap each other. Those five should sum close to this value; "
        "the remainder is result regrouping, projecting results back to the "
        "source camera, and optional per-frame NPZ writes. Exception: with "
        "--frame_batch_size, smoothing runs once per clip after all batches, "
        "outside this stage (still inside total_wall)."
    ),
    PerfStage.PREPROCESSING: (
        "Per-frame wall time of hand-crop preparation: either perspective "
        "cropping (--perspective_cropping) or bbox cropping + resizing, "
        "whichever mode this run uses -- the two are mutually exclusive "
        "per run, never both."
    ),
    PerfStage.BATCH_ASSEMBLY: (
        "Per-frame wall time of stacking prepared crops and camera data into "
        "batch tensors and copying them to the GPU, up to and including a "
        "torch.cuda.synchronize() that waits for those copies to finish."
    ),
    PerfStage.MODEL_FORWARD: (
        "Per-frame wall time of the model's forward pass only. Bracketed "
        "with torch.cuda.synchronize() before and after, so it is an "
        "accurate GPU-inclusive number and none of its GPU time leaks into "
        "post_forward."
    ),
    PerfStage.POST_FORWARD: (
        "Per-frame wall time of everything after the forward pass for each "
        "batch: camera translation/projection math, the right-hand mirror, "
        "copying outputs back to the CPU, and assembling per-crop result "
        "dicts."
    ),
    PerfStage.ADAPTIVE_SMOOTHING: (
        "Per-frame wall time of adaptive SOMA parameter smoothing and "
        "geometry reconstruction. Only recorded when --postprocessing_smooth "
        "is on (the default)."
    ),
    PerfStage.OVERLAY_RENDERING: (
        "Per-frame wall time of render_clip(...): per-frame skeleton/bbox/"
        "mesh drawing plus video encoding (bbox_viz.mp4, fullimg_overlay.mp4). "
        "Not recorded with --no_save_videos or --frame_batch_size, since "
        "overlays are skipped entirely in both cases."
    ),
    PerfStage.TOTAL_WALL: (
        "Per-frame wall time of the whole per-video processing block -- "
        "bbox detection through overlay rendering, everything this script "
        "does for one video after the model is loaded. Useful for sanity-"
        "checking that the other stages plus model_load roughly account "
        "for the full run; the remaining gap is uninstrumented work such as "
        "video decoding, NPZ writing, and per-video setup."
    ),
}


@runtime_checkable
class PerfReporter(Protocol):
    def record_latency(self, stage: PerfStage, latency_ms: float) -> None: ...


class LocalPerfWriter:
    """Accumulates per-stage latencies and writes them to a JSON file.

    Intended to share the lifecycle of the inference entrypoint: constructed
    once, receives one ``record_latency()`` call per stage per inference run,
    and ``write()`` is called once at the end to persist results.

    Output schema::

        {
            "metadata": {"checkpoint": "...", "device": "...", "dtype": "..."},
            "model_load_ms": 20085.16,
            "model_load_description": "...",
            "frames": [600],
            "stages": {
                "bbox_detection": {
                    "description": "...",
                    "samples": [44.2, 41.8, ...],
                    "avg_ms": 39.97,
                    "count": 2
                },
                "inference": {
                    "description": "...",
                    "samples": [57.2, 55.9, ...],
                    "avg_ms": 56.55,
                    "count": 2
                }
            }
        }

    Each stage's ``description`` (and the top-level ``model_load_description``)
    comes from ``STAGE_DESCRIPTIONS``, so the file is self-explanatory without
    cross-referencing this module or ``inference/perf/README.md``.
    """

    def __init__(self, output_path: str | Path, metadata: dict | None = None) -> None:
        self._output_path = Path(output_path)
        self._metadata = metadata or {}
        self._latencies: dict[PerfStage, list[float]] = {}
        self._frames = []

    def record_latency(self, stage: PerfStage, latency_ms: float) -> None:
        self._latencies.setdefault(stage, []).append(latency_ms)

    def record_frames(self, frames: int) -> None:
        self._frames.append(frames)

    def write(self) -> Path:
        """Write accumulated latencies to JSON and return the output path."""
        model_load = self._latencies.get(PerfStage.MODEL_LOAD)
        stages = {}
        for stage, values in self._latencies.items():
            if stage is PerfStage.MODEL_LOAD:
                continue
            stages[stage.value] = {
                "description": STAGE_DESCRIPTIONS[stage],
                "samples": values,
                "avg_ms": sum(values) / len(values),
                "count": len(values),
            }
        record = {
            "metadata": self._metadata,
            "model_load_ms": model_load[0] if model_load else None,
            "model_load_description": STAGE_DESCRIPTIONS[PerfStage.MODEL_LOAD],
            "frames": self._frames,
            "stages": stages,
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(json.dumps(record, indent=2))
        return self._output_path

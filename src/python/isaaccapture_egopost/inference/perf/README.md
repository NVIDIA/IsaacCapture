<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
-->

# inference/perf

Lightweight performance instrumentation for the EgoHand inference pipeline.

## What is measured

Each stage reports wall-clock elapsed time in milliseconds, measured with `time.perf_counter()`. Per-frame stages are the stage's total time divided by the number of frames it processed, so each value is an average ms per frame. `MODEL_LOAD` is a one-time total, not per frame. `MODEL_FORWARD` is bracketed with CUDA synchronization, so it includes GPU execution time.

| Stage | What is measured |
|---|---|
| `MODEL_LOAD` | One-time: `EgoHandPredictor.from_checkpoint(...)` (checkpoint load + model construction + device placement) |
| `BBOX_DETECTION` | `pipeline.bbox_detector.run(...)` -- whichever backend is configured (`rtdetr`, or the JSON-read time for `saved_bboxes`). Not recorded when `bbox_detector` is `None` (ground-truth boxes passed directly) |
| `INFERENCE` | The whole `predictor.run(...)` call. Roughly the sum of `PREPROCESSING` + `BATCH_ASSEMBLY` + `MODEL_FORWARD` + `POST_FORWARD` + `ADAPTIVE_SMOOTHING` |
| `PREPROCESSING` | Hand-crop preparation: perspective cropping (`--perspective_cropping`) or bbox cropping + resizing, whichever mode the run uses (never both) |
| `BATCH_ASSEMBLY` | Stacking crops and camera data into batch tensors and copying them to the GPU |
| `MODEL_FORWARD` | `model.forward_step(...)` only, bracketed with `torch.cuda.synchronize()` so its GPU time doesn't leak into neighboring stages |
| `POST_FORWARD` | Camera math, right-hand mirror, copying outputs back to the CPU, and assembling result dicts |
| `ADAPTIVE_SMOOTHING` | `smooth_adaptive_hand_results(...)`. Only recorded when `--postprocessing_smooth` is on (the default) |
| `OVERLAY_RENDERING` | `render_clip(...)`: overlay drawing plus video encoding. Not recorded with `--no_save_videos` or `--frame_batch_size` |
| `TOTAL_WALL` | The whole per-video block in `process_video(...)`, from bbox detection through overlay rendering |

How the stages nest (siblings run one after another and never overlap):

```text
MODEL_LOAD                     (once, before the per-video loop)
TOTAL_WALL                     (per video)
├── BBOX_DETECTION
├── INFERENCE
│   ├── PREPROCESSING
│   ├── BATCH_ASSEMBLY  ┐
│   ├── MODEL_FORWARD   ├ per batch of --batch_size crops
│   ├── POST_FORWARD    ┘
│   └── ADAPTIVE_SMOOTHING
└── OVERLAY_RENDERING
```

Only `INFERENCE` and `TOTAL_WALL` contain other stages. The children of each parent sum close to it, but not exactly: `INFERENCE` also includes result regrouping, projection back to the source camera, and optional per-frame NPZ writes, and `TOTAL_WALL` also includes video decoding, NPZ writing, and per-video setup. One exception: with `--frame_batch_size`, smoothing runs once per clip after all batches, so `ADAPTIVE_SMOOTHING` moves out of `INFERENCE` but stays inside `TOTAL_WALL`.

`MODEL_LOAD` is measured once per run. The per-frame stages are measured once per pipeline call -- once per video unless `--frame_batch_size` splits it into multiple batches (see `inference/scripts/README.md`). `TOTAL_WALL` and `OVERLAY_RENDERING` are measured once per video.

Timing synchronizes the GPU around the forward pass, which adds a small overhead compared with an uninstrumented run.

## Output format

Results are written per video to `<output_dir>/<video_stem>/perf.json` (or to a single file with `--perf_output`).

```jsonc
{
  "metadata": {"checkpoint": "ckpt_v2", "device": "NVIDIA RTX 6000 Ada", "dtype": "float32", "batch_size": 32},
  "model_load_ms": 20085.16,
  "model_load_description": "One-time cost: checkpoint load, model construction, ...",
  "frames": [600],
  "stages": {
    "model_forward": {
      "description": "Per-frame wall time of the model's forward pass only, ...",
      "samples": [13.6, 13.5, 13.7],
      "avg_ms": 13.6,
      "count": 3
    },
    "inference": {
      "description": "Per-frame wall time of the whole predictor call, ...",
      "samples": [57.2, 55.9, 84.8],
      "avg_ms": 65.97,
      "count": 3
    }
    // other stages, same shape: bbox_detection, preprocessing, batch_assembly,
    // post_forward, adaptive_smoothing, overlay_rendering, total_wall
  }
}
```

`frames` lists the frame count of each batch processed, in order -- one entry per video for a typical run; more than one if `--frame_batch_size` split a video into multiple batches.

`model_load_ms` is a top-level scalar (one-time cost, not per-frame). Each stage under `stages` has the raw per-run samples alongside the pre-computed average and count. `MODEL_LOAD` is never included under `stages`.

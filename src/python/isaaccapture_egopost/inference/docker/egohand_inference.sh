#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

# Local helper for building and running the egohand-inference Docker image.
# Run from the EgoHand repo root, or any subdirectory — the script locates the root itself.
# Run with no arguments for usage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.egohand"

# Repo-relative paths staged into the build context. Keep in sync with the
# Dockerfile's COPY lines — anything listed there must be listed here.
CONTEXT_PATHS=(
  egohand
  inference
  requirements.txt
  requirements-dinov3.txt
  requirements
)

# Fixed container-side paths for bundled and volume-mounted weights.
C_EGOHAND_CKPT_BUNDLED=/workspace/weights/egohand.ckpt
C_RTDETR_CKPT_BUNDLED=/workspace/weights/rtdetr.onnx
C_INPUT_DIR=/workspace/inputs/video
C_OUTPUT=/workspace/outputs/inference

usage() {
  cat >&2 <<EOF
usage: inference/docker/egohand_inference.sh <build|run> [args]

  build <image> [--egohand-ckpt <path>] [--hand-detector-weight <path>]
      Build the named Docker image from inference/docker/Dockerfile.egohand, with the
      source baked in. First build: ~15 min (pip deps). Re-builds after
      source-only changes: <1 min (the pip layer is cached; only COPY layers
      re-run).

      <image>                      Docker image name/tag to build. Required.
      --egohand-ckpt <path>        Bundle an EgoHand checkpoint into the image at
                                   $C_EGOHAND_CKPT_BUNDLED. Optional — if omitted,
                                   pass --ckpt at run time, or omit both to
                                   resolve via HF (nvidia/EgoHand-v1) instead.
      --hand-detector <rtdetr|saved_bboxes>
                                   Which detector the bundled weight is for
                                   (default: rtdetr, only option). Determines the
                                   container path: rtdetr → $C_RTDETR_CKPT_BUNDLED.
                                   saved_bboxes needs no detector weight (bbox
                                   detection is precomputed and supplied at run
                                   time via --bbox-dir); --hand-detector-weight
                                   is rejected with this value.
      --hand-detector-weight <path> Bundle the hand detector weight into the
                                   image. Optional — if omitted, pass
                                   --hand-detector-weight at run time, or omit
                                   both to resolve via HF
                                   (nvidia/RT-DETR-Hand-Detector-v1) instead.

  run --image <name> --input <path> --intrinsics-json <path> [--ckpt <path>] [options]
      Run EgoHand hand inference through the image entrypoint,
      inference/scripts/run_egohand_inference_pipeline.py — hand bbox detection
      feeding inference/pipeline/pipeline.py:EgoHandInferencePipeline.

      --image <name>          Docker image name/tag to run. Required.
      --ckpt                  EgoHand hand checkpoint (.ckpt), mounted read-only.
                              Not required if the image was built with --egohand-ckpt.
                              If neither is given, resolved via HF
                              (nvidia/EgoHand-v1) at run time.
      --input                 a video file, or a folder of .mp4 files; mounted read-only.
      --hand-detector         Hand detector backend: rtdetr (default, only option).
                              Ignored if --bbox-dir is passed.
      --hand-detector-weight  Detector weight file (.onnx for rtdetr), mounted read-only.
                              Not required if the image was built with --hand-detector-weight.
                              If neither is given, resolved via HF
                              (nvidia/RT-DETR-Hand-Detector-v1) at run time.
                              Rejected together with --bbox-dir.
      --bbox-dir <path>       Skip on-the-fly bbox detection and read precomputed
                              per-hand bbox JSONs instead, mounted read-only.
                              Layout must match inference/hand_detector_2d/scripts/
                              run_hand_detector.py's --output_dir:
                              <bbox-dir>/<video_stem>/<video_stem>_{left,right}_bbox.json.
                              A saved_bboxes detector config pointing at the mounted
                              path is generated per run (detector_backend: saved_bboxes).
      --intrinsics-json       Required by this wrapper. Camera intrinsics JSON, mounted
                              read-only. This model needs the real calibration to produce
                              correct depth/mesh scale.
      --perspective-cropping  Create hand crops directly from the source camera.
                              Required with --bbox-dir for distorted calibration.
                              Default: off (undistort full frames before cropping).
      --hf-token <token>       Optional. Forwarded as HF_TOKEN in the container
                              for authenticated HF downloads (e.g. higher rate
                              limits). Not needed for normal runs.
      --out <path>            Output dir (default: outputs/inference). Per-video results
                              land in <out>/<video_stem>/.
      Hardcoded: --hand both --device cuda --batch_size 32 --dtype bfloat16
                 --draw_skeleton --no_mesh
EOF
  exit 1
}

die() { echo "error: $*" >&2; exit 1; }

cmd="${1:-}"
shift || true

case "$cmd" in

  build)
    [[ $# -gt 0 && "$1" != --* ]] || { echo "error: image name is required as the first argument to build" >&2; usage; }
    IMAGE="$1"; shift
    EGOHAND_CKPT_PATH=""
    HAND_DETECTOR="rtdetr"
    HAND_DETECTOR_PATH=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --egohand-ckpt)          EGOHAND_CKPT_PATH="$2";   shift 2 ;;
        --hand-detector)         HAND_DETECTOR="$2";       shift 2 ;;
        --hand-detector-weight)  HAND_DETECTOR_PATH="$2"; shift 2 ;;
        *) echo "unknown build arg: $1" >&2; usage ;;
      esac
    done
    [[ "$HAND_DETECTOR" == "rtdetr" || "$HAND_DETECTOR" == "saved_bboxes" ]] \
      || die "--hand-detector must be rtdetr or saved_bboxes (got: $HAND_DETECTOR)"
    [[ "$HAND_DETECTOR" != "saved_bboxes" || -z "$HAND_DETECTOR_PATH" ]] \
      || die "--hand-detector-weight is not applicable with --hand-detector saved_bboxes (bbox detection is precomputed, no detector weight is loaded)"

    BUILD_CTX="$(mktemp -d)"
    trap 'rm -rf "$BUILD_CTX"' EXIT

    echo "=== staging build context → $BUILD_CTX ==="
    for path in "${CONTEXT_PATHS[@]}"; do
      [[ -e "$REPO_ROOT/$path" ]] || die "missing from repo: $path"
      mkdir -p "$BUILD_CTX/$(dirname "$path")"
      cp -r "$REPO_ROOT/$path" "$BUILD_CTX/$path"
    done

    # Always create weights/ so the Dockerfile COPY is valid even without weights.
    mkdir -p "$BUILD_CTX/weights"

    if [[ -n "$EGOHAND_CKPT_PATH" ]]; then
      [[ -f "$EGOHAND_CKPT_PATH" ]] || die "EgoHand checkpoint not found: $EGOHAND_CKPT_PATH"
      echo "=== bundling EgoHand checkpoint: $EGOHAND_CKPT_PATH ==="
      cp "$EGOHAND_CKPT_PATH" "$BUILD_CTX/weights/egohand.ckpt"
    fi

    if [[ -n "$HAND_DETECTOR_PATH" ]]; then
      [[ -f "$HAND_DETECTOR_PATH" ]] || die "hand detector weight not found: $HAND_DETECTOR_PATH"
      echo "=== bundling hand detector weight ($HAND_DETECTOR): $HAND_DETECTOR_PATH ==="
      cp "$HAND_DETECTOR_PATH" "$BUILD_CTX/weights/rtdetr.onnx"
    fi

    echo "=== building $IMAGE ==="
    # COPY keeps the staged modes, and the image runs as a non-root user, so
    # make everything world-readable in case the host umask (e.g. 027) wasn't.
    chmod -R a+rX "$BUILD_CTX"
    docker build -t "$IMAGE" -f "$DOCKERFILE" "$BUILD_CTX"
    echo "=== build done ==="
    ;;

  run)
    IMAGE=""
    CKPT=""
    INPUT=""
    HAND_DETECTOR="rtdetr"
    HAND_DETECTOR_WEIGHT=""
    BBOX_DIR_PATH=""
    INTRINSICS_JSON=""
    CROP_ARGS=()
    HF_TOKEN_ARG=""
    OUT="$REPO_ROOT/outputs/inference"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --image)                 IMAGE="$2";                shift 2 ;;
        --ckpt)                  CKPT="$2";                 shift 2 ;;
        --input)                 INPUT="$2";                shift 2 ;;
        --hand-detector)         HAND_DETECTOR="$2";        shift 2 ;;
        --hand-detector-weight)  HAND_DETECTOR_WEIGHT="$2"; shift 2 ;;
        --bbox-dir)              BBOX_DIR_PATH="$2";        shift 2 ;;
        --intrinsics-json)       INTRINSICS_JSON="$2";      shift 2 ;;
        --perspective-cropping)  CROP_ARGS=(--perspective_cropping); shift ;;
        --hf-token)              HF_TOKEN_ARG="$2";         shift 2 ;;
        --out)                   OUT="$2";                  shift 2 ;;
        *) echo "unknown arg: $1" >&2; usage ;;
      esac
    done

    [[ -n "$IMAGE" ]] || die "--image is required"
    [[ -n "$INPUT" ]] || die "--input is required"
    [[ -e "$INPUT" ]] || die "input not found: $INPUT"
    # --intrinsics-json is required -- the model needs the real calibration to produce correct depth/mesh scale
    [[ -n "$INTRINSICS_JSON" ]] \
      || die "--intrinsics-json is required -- the model needs the real calibration to produce correct depth/mesh scale"
    [[ "$HAND_DETECTOR" == "rtdetr" ]] \
      || die "--hand-detector must be rtdetr (got: $HAND_DETECTOR)"
    [[ -z "$BBOX_DIR_PATH" || -z "$HAND_DETECTOR_WEIGHT" ]] \
      || die "--bbox-dir and --hand-detector-weight are mutually exclusive (--bbox-dir skips detection entirely, so no detector weight is loaded)"

    INPUT="$(realpath "$INPUT")"
    OUT="$(realpath -m "$OUT")"
    mkdir -p "$OUT"

    VOLUME_ARGS=()
    RUN_EGOHAND_CKPT=""
    RUN_INTRINSICS=""
    RUN_HAND_DETECTOR_CKPT=""

    # Mount EgoHand checkpoint if provided; else use it if the image was built
    # with --egohand-ckpt; else leave RUN_EGOHAND_CKPT empty and omit --egohand_ckpt
    # entirely, so run_egohand_inference_pipeline.py falls back to its own HF
    # auto-download (nvidia/EgoHand-v1) instead of failing on a bundled
    # path that was never populated.
    if [[ -n "$CKPT" ]]; then
      [[ -f "$CKPT" ]] || die "checkpoint not found: $CKPT"
      CKPT="$(realpath "$CKPT")"
      C_EGOHAND_CKPT_MOUNTED="/workspace/inputs/$(basename "$CKPT")"
      VOLUME_ARGS+=(-v "$CKPT":"$C_EGOHAND_CKPT_MOUNTED":ro)
      RUN_EGOHAND_CKPT="$C_EGOHAND_CKPT_MOUNTED"
    # Probe the image itself: the host has no way to know whether this
    # image was actually built with a bundled checkpoint (COPY weights/ is
    # always valid, bundled or not -- see egohand_inference.sh build).
    elif docker run --rm "$IMAGE" test -f "$C_EGOHAND_CKPT_BUNDLED" 2>/dev/null; then
      RUN_EGOHAND_CKPT="$C_EGOHAND_CKPT_BUNDLED"
    fi

    # Required (checked above) to produce correct depth/mesh scale -- always
    # mounted here.
    if [[ -n "$INTRINSICS_JSON" ]]; then
      [[ -f "$INTRINSICS_JSON" ]] || die "intrinsics JSON not found: $INTRINSICS_JSON"
      INTRINSICS_JSON="$(realpath "$INTRINSICS_JSON")"
      C_INTRINSICS_MOUNTED="/workspace/inputs/$(basename "$INTRINSICS_JSON")"
      VOLUME_ARGS+=(-v "$INTRINSICS_JSON":"$C_INTRINSICS_MOUNTED":ro)
      RUN_INTRINSICS="$C_INTRINSICS_MOUNTED"
    fi

    # Mount hand detector weight if provided; else use it if the image was
    # built with --hand-detector-weight; else leave RUN_HAND_DETECTOR_CKPT
    # empty, which leaves the generated config's weights: blank (null) --
    # inference/hand_detector_2d/configs/rtdetr.py falls back to HF
    # (nvidia/RT-DETR-Hand-Detector-v1) when weights is unset. Skipped
    # entirely for --bbox-dir runs, which never load a detector weight.
    if [[ -z "$BBOX_DIR_PATH" ]]; then
      if [[ -n "$HAND_DETECTOR_WEIGHT" ]]; then
        [[ -f "$HAND_DETECTOR_WEIGHT" ]] || die "hand detector weight not found: $HAND_DETECTOR_WEIGHT"
        HAND_DETECTOR_WEIGHT="$(realpath "$HAND_DETECTOR_WEIGHT")"
        C_HAND_DETECTOR_CKPT_MOUNTED="/workspace/inputs/$(basename "$HAND_DETECTOR_WEIGHT")"
        VOLUME_ARGS+=(-v "$HAND_DETECTOR_WEIGHT":"$C_HAND_DETECTOR_CKPT_MOUNTED":ro)
        RUN_HAND_DETECTOR_CKPT="$C_HAND_DETECTOR_CKPT_MOUNTED"
      # Probe the image itself: the host has no way to know whether this
      # image was actually built with a bundled weight (COPY weights/ is
      # always valid, bundled or not -- see egohand_inference.sh build).
      elif docker run --rm "$IMAGE" test -f "$C_RTDETR_CKPT_BUNDLED" 2>/dev/null; then
        RUN_HAND_DETECTOR_CKPT="$C_RTDETR_CKPT_BUNDLED"
      fi
    fi

    if [[ -f "$INPUT" ]]; then
      C_INPUT="$C_INPUT_DIR/$(basename "$INPUT")"
    else
      C_INPUT="$C_INPUT_DIR"
    fi

    # Mount precomputed bbox dir if provided; overrides on-the-fly detection.
    if [[ -n "$BBOX_DIR_PATH" ]]; then
      [[ -d "$BBOX_DIR_PATH" ]] || die "bbox dir not found: $BBOX_DIR_PATH"
      BBOX_DIR_PATH="$(realpath "$BBOX_DIR_PATH")"
      C_BBOX_DIR="/workspace/inputs/bbox"
      VOLUME_ARGS+=(-v "$BBOX_DIR_PATH":"$C_BBOX_DIR":ro)

      # Copied from the repo's saved_bboxes.yaml rather than hand-written, so any
      # other fields added there stay in sync automatically; only saved_bbox_path
      # is overridden. Rewrites the whole saved_bbox_path: line rather than
      # substituting the ${BBOX_DIR} placeholder specifically, so this doesn't
      # silently no-op (leaving a host path baked into the container's config)
      # if that file's local copy ever has saved_bbox_path set to something
      # else, e.g. a hardcoded path from manual host-side testing.
      CONFIG_DIR="$(mktemp -d)"
      trap 'rm -rf "$CONFIG_DIR"' EXIT
      CONFIG_FILE="$CONFIG_DIR/saved_bboxes.yaml"
      cp "$REPO_ROOT/inference/hand_detector_2d/configs/saved_bboxes.yaml" "$CONFIG_FILE"
      sed -i "s#^saved_bbox_path:.*#saved_bbox_path: \"$C_BBOX_DIR\"#" "$CONFIG_FILE"
      C_CONFIG="/workspace/inputs/saved_bboxes.yaml"
      VOLUME_ARGS+=(-v "$CONFIG_FILE":"$C_CONFIG":ro)
      DETECTOR_ARGS=(--bbox_detector_config "$C_CONFIG")
    else
      # Copied from the repo's rtdetr_docker.yaml rather than hand-written, so any
      # other fields set there (score_thr, padding, tracking, ...) stay in sync
      # automatically; only the bundled weight path is overridden. Using the
      # static file directly, unmodified, would silently ignore a
      # --hand-detector-weight override -- it always points at
      # $C_RTDETR_CKPT_BUNDLED. When RUN_HAND_DETECTOR_CKPT is empty (neither
      # mounted nor bundled), this substitutes an empty value, leaving
      # weights: blank (null) so rtdetr.py's config builder falls back to HF.
      CONFIG_DIR="$(mktemp -d)"
      trap 'rm -rf "$CONFIG_DIR"' EXIT
      CONFIG_FILE="$CONFIG_DIR/hand_detector.yaml"
      cp "$REPO_ROOT/inference/hand_detector_2d/configs/rtdetr_docker.yaml" "$CONFIG_FILE"
      sed -i "s#$C_RTDETR_CKPT_BUNDLED#$RUN_HAND_DETECTOR_CKPT#" "$CONFIG_FILE"
      C_CONFIG="/workspace/inputs/hand_detector.yaml"
      VOLUME_ARGS+=(-v "$CONFIG_FILE":"$C_CONFIG":ro)
      DETECTOR_ARGS=(--bbox_detector_config "$C_CONFIG")
    fi

    echo "=== running $IMAGE ==="
    if [[ -n "$CKPT" ]]; then
      echo "    egohand_ckpt      : $CKPT (mounted)"
    elif [[ -n "$RUN_EGOHAND_CKPT" ]]; then
      echo "    egohand_ckpt      : bundled ($RUN_EGOHAND_CKPT)"
    else
      echo "    egohand_ckpt      : none -- resolved via HF (nvidia/EgoHand-v1) at run time"
    fi
    if [[ -n "$BBOX_DIR_PATH" ]]; then
      echo "    hand_detector : saved_bboxes (precomputed, $BBOX_DIR_PATH)"
    elif [[ -n "$HAND_DETECTOR_WEIGHT" ]]; then
      echo "    hand_detector : $HAND_DETECTOR ($HAND_DETECTOR_WEIGHT, mounted)"
    elif [[ -n "$RUN_HAND_DETECTOR_CKPT" ]]; then
      echo "    hand_detector : $HAND_DETECTOR (bundled, $RUN_HAND_DETECTOR_CKPT)"
    else
      echo "    hand_detector : $HAND_DETECTOR (none -- resolved via HF (nvidia/RT-DETR-Hand-Detector-v1) at run time)"
    fi
    echo "    intrinsics    : $RUN_INTRINSICS"
    echo "    hf_token      : ${HF_TOKEN_ARG:+set}"
    echo "    input         : $INPUT"
    echo "    out           : $OUT"

    # Run as the calling user so files written to the mounted output dir are
    # owned by them on the host, not by the image's user.
    docker run --rm --gpus all \
      --user "$(id -u):$(id -g)" \
      -e PYTHONUNBUFFERED=1 \
      ${HF_TOKEN_ARG:+-e HF_TOKEN="$HF_TOKEN_ARG"} \
      "${VOLUME_ARGS[@]+"${VOLUME_ARGS[@]}"}" \
      -v "$INPUT":"$C_INPUT":ro \
      -v "$OUT":"$C_OUTPUT" \
      "$IMAGE" \
      python inference/scripts/run_egohand_inference_pipeline.py \
        --input "$C_INPUT" \
        ${RUN_EGOHAND_CKPT:+--egohand_ckpt "$RUN_EGOHAND_CKPT"} \
        --output_dir "$C_OUTPUT" \
        --intrinsics_json "$RUN_INTRINSICS" \
        --hand both \
        --device cuda \
        --batch_size 32 \
        --dtype bfloat16 \
        --draw_skeleton --no_mesh \
        "${CROP_ARGS[@]}" \
        "${DETECTOR_ARGS[@]}"
    ;;

  *)
    usage
    ;;

esac

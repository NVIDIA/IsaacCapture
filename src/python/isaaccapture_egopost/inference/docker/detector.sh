#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

# Local helper for building and running the standalone RT-DETRv2 bbox-detector
# Docker image (inference/docker/Dockerfile.detector). Detection only -- no EgoHand
# model, no SOMA. Pair with egohand_inference.sh's --bbox-dir to feed the JSONs
# this produces into a full EgoHand inference run without re-detecting.
# Run from the EgoHand repo root, or any subdirectory -- the script locates the root itself.
# Run with no arguments for usage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.detector"

C_WEIGHTS_BUNDLED=/workspace/weights/rtdetr.onnx
C_INPUT_DIR=/workspace/inputs/video
C_OUTPUT=/workspace/outputs/bbox

usage() {
  cat >&2 <<EOF
usage: inference/docker/detector.sh <build|run> [args]

  build <image> [--weight <path>]
      Build the named Docker image from inference/docker/Dockerfile.detector,
      with inference/hand_detector_2d baked in. No EgoHand source, no torch usage --
      only numpy/opencv/pillow/pyyaml/onnxruntime-gpu.

      <image>            Docker image name/tag to build. Required.
      --weight <path>    Bundle an RT-DETRv2 ONNX weight into the image at
                          $C_WEIGHTS_BUNDLED. Optional -- if omitted,
                          pass --weight at run time, or omit both to resolve
                          via HF (nvidia/RT-DETR-Hand-Detector-v1) instead.

  run --image <name> --input <path> --out <path> --intrinsics-json <path> [options]
      Run inference/hand_detector_2d/scripts/run_hand_detector.py through the
      image entrypoint: bbox detection only, written as
      <out>/<video_stem>/<video_stem>_{left,right}_bbox.json -- the layout
      egohand_inference.sh run --bbox-dir <out> expects.

      Use the same input video and --intrinsics-json for the paired
      egohand_inference.sh run. This wrapper uses the detector's default
      pixel-aspect rectification and does not expose a switch to disable it.
      For standard pinhole cropping, match both passes' pixel-aspect settings
      to reproduce live detector preprocessing. Perspective cropping skips
      full-frame rectification; to match live detection in that mode, use
      run_hand_detector.py directly with --no-rectify_pixel_aspect_ratio.
      Rectification can change detections when it resizes the images, but
      saved boxes are always mapped back to source coordinates.

      --image <name>       Docker image name/tag to run. Required.
      --input <path>       A video file, or a folder of .mp4 files; mounted read-only.
      --out <path>         Output dir for bbox JSONs. Required.
      --intrinsics-json <path>
                            Camera intrinsics JSON, mounted read-only. Required --
                            must match the paired egohand_inference.sh run's
                            --intrinsics-json (see above).
      --weight <path>      RT-DETRv2 ONNX weight, mounted read-only. Not required
                            if the image was built with --weight -- and if neither
                            is given, resolved via HF (nvidia/RT-DETR-Hand-Detector-v1)
                            at run time instead.
      --hand <left|right|both>  Default: both.
      --tracking / --no-tracking
                            Override tracking from the shared rtdetr_docker.yaml
                            (currently on). --no-tracking also disables bbox smoothing.
      --device <dev>        Torch-style device, e.g. cuda:1 or cpu. Default: cuda.
      --max-seconds <N>      Only process the first N seconds.
      --frame-batch-size <N> Frames decoded into RAM at once. Default: 2000.
      --hf-token <token>     Optional. Forwarded as HF_TOKEN in the container
                              for authenticated HF downloads (e.g. higher rate
                              limits). Not needed for normal runs.
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
    WEIGHT_PATH=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --weight) WEIGHT_PATH="$2"; shift 2 ;;
        *) echo "unknown build arg: $1" >&2; usage ;;
      esac
    done

    BUILD_CTX="$(mktemp -d)"
    trap 'rm -rf "$BUILD_CTX"' EXIT

    echo "=== staging build context -> $BUILD_CTX ==="
    mkdir -p "$BUILD_CTX/inference"
    cp -r "$REPO_ROOT/inference/hand_detector_2d" "$BUILD_CTX/inference/hand_detector_2d"

    # Always create weights/ so the Dockerfile COPY is valid even without a weight.
    mkdir -p "$BUILD_CTX/weights"
    if [[ -n "$WEIGHT_PATH" ]]; then
      [[ -f "$WEIGHT_PATH" ]] || die "weight not found: $WEIGHT_PATH"
      echo "=== bundling weight: $WEIGHT_PATH ==="
      cp "$WEIGHT_PATH" "$BUILD_CTX/weights/rtdetr.onnx"
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
    INPUT=""
    OUT=""
    INTRINSICS_JSON=""
    WEIGHT=""
    HAND="both"
    TRACKING=""
    DEVICE="cuda"
    MAX_SECONDS=""
    FRAME_BATCH_SIZE="2000"
    HF_TOKEN_ARG=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --image)             IMAGE="$2";            shift 2 ;;
        --input)             INPUT="$2";            shift 2 ;;
        --out)                OUT="$2";             shift 2 ;;
        --intrinsics-json)    INTRINSICS_JSON="$2";  shift 2 ;;
        --weight)             WEIGHT="$2";          shift 2 ;;
        --hand)                HAND="$2";           shift 2 ;;
        --tracking)            TRACKING="true";     shift 1 ;;
        --no-tracking)         TRACKING="false";    shift 1 ;;
        --device)              DEVICE="$2";         shift 2 ;;
        --max-seconds)         MAX_SECONDS="$2";    shift 2 ;;
        --frame-batch-size)    FRAME_BATCH_SIZE="$2"; shift 2 ;;
        --hf-token)            HF_TOKEN_ARG="$2";    shift 2 ;;
        *) echo "unknown arg: $1" >&2; usage ;;
      esac
    done

    [[ -n "$IMAGE" ]] || die "--image is required"
    [[ -n "$INPUT" ]] || die "--input is required"
    [[ -e "$INPUT" ]] || die "input not found: $INPUT"
    [[ -n "$OUT" ]] || die "--out is required"
    [[ -n "$INTRINSICS_JSON" ]] \
      || die "--intrinsics-json is required -- must match the paired egohand_inference.sh run's --intrinsics-json, or the boxes land on the wrong pixels (see usage)"
    [[ -f "$INTRINSICS_JSON" ]] || die "intrinsics JSON not found: $INTRINSICS_JSON"
    [[ "$HAND" == "left" || "$HAND" == "right" || "$HAND" == "both" ]] \
      || die "--hand must be left, right, or both (got: $HAND)"

    INPUT="$(realpath "$INPUT")"
    OUT="$(realpath -m "$OUT")"
    mkdir -p "$OUT"
    INTRINSICS_JSON="$(realpath "$INTRINSICS_JSON")"

    VOLUME_ARGS=()
    C_INTRINSICS_MOUNTED="/workspace/inputs/$(basename "$INTRINSICS_JSON")"
    VOLUME_ARGS+=(-v "$INTRINSICS_JSON":"$C_INTRINSICS_MOUNTED":ro)
    RUN_WEIGHT=""
    # When neither --weight nor a bundled weight is used, RUN_WEIGHT stays
    # empty and the generated config's weights: below is left blank --
    # run_hand_detector.py's load_rtdetr_config()
    # falls back to HF (nvidia/RT-DETR-Hand-Detector-v1) when weights is
    # unset, same as inference/hand_detector_2d/configs/rtdetr.py's build()
    # does for the full pipeline's rtdetr backend.
    if [[ -n "$WEIGHT" ]]; then
      [[ -f "$WEIGHT" ]] || die "weight not found: $WEIGHT"
      WEIGHT="$(realpath "$WEIGHT")"
      C_WEIGHT_MOUNTED="/workspace/inputs/$(basename "$WEIGHT")"
      VOLUME_ARGS+=(-v "$WEIGHT":"$C_WEIGHT_MOUNTED":ro)
      RUN_WEIGHT="$C_WEIGHT_MOUNTED"
    # Probe the image itself: the host has no way to know whether this
    # image was actually built with a bundled weight (COPY weights/ is
    # always valid, bundled or not -- see detector.sh build).
    elif docker run --rm "$IMAGE" test -f "$C_WEIGHTS_BUNDLED" 2>/dev/null; then
      RUN_WEIGHT="$C_WEIGHTS_BUNDLED"
    fi
    # Neither mounted nor bundled: RUN_WEIGHT stays empty, so the generated
    # config's weights: is left blank (null) -- rtdetr.py's config builder
    # falls back to fetching from HF (nvidia/RT-DETR-Hand-Detector-v1) when
    # weights is unset, same as run_egohand_inference_pipeline.py's --egohand_ckpt.

    if [[ -f "$INPUT" ]]; then
      C_INPUT="$C_INPUT_DIR/$(basename "$INPUT")"
    else
      C_INPUT="$C_INPUT_DIR"
    fi

    # Inherit shared detector settings, including bbox smoothing. Override only
    # container-specific weights and tracking when explicitly requested.
    CONFIG_DIR="$(mktemp -d)"
    trap 'rm -rf "$CONFIG_DIR"' EXIT
    CONFIG_FILE="$CONFIG_DIR/rtdetr.yaml"
    cp "$REPO_ROOT/inference/hand_detector_2d/configs/rtdetr_docker.yaml" "$CONFIG_FILE"
    # Quote paths for YAML, then escape sed replacement characters. An empty
    # weight selects the existing HuggingFace fallback.
    WEIGHT_YAML="null"
    if [[ -n "$RUN_WEIGHT" ]]; then
      WEIGHT_YAML="'${RUN_WEIGHT//\'/\'\'}'"
    fi
    WEIGHT_REPLACEMENT="$(printf '%s' "$WEIGHT_YAML" | sed 's/[\\&#]/\\&/g')"
    sed -i "s#^  weights:.*#  weights: $WEIGHT_REPLACEMENT#" "$CONFIG_FILE"
    if [[ -n "$TRACKING" ]]; then
      sed -i "s/^tracking:.*/tracking: $TRACKING/" "$CONFIG_FILE"
    fi
    if [[ "$TRACKING" == "false" ]]; then
      sed -i 's/^  bbox_smoothing:.*/  bbox_smoothing: false/' "$CONFIG_FILE"
    fi
    C_CONFIG="/workspace/inputs/rtdetr.yaml"
    VOLUME_ARGS+=(-v "$CONFIG_FILE":"$C_CONFIG":ro)

    echo "=== running $IMAGE ==="
    if [[ -n "$WEIGHT" ]]; then
      echo "    weight   : $WEIGHT (mounted)"
    elif [[ -n "$RUN_WEIGHT" ]]; then
      echo "    weight   : bundled ($RUN_WEIGHT)"
    else
      echo "    weight   : none -- resolved via HF (nvidia/RT-DETR-Hand-Detector-v1) at run time"
    fi
    echo "    hand     : $HAND"
    echo "    tracking : ${TRACKING:-from shared config}"
    echo "    hf_token : ${HF_TOKEN_ARG:+set}"
    echo "    input    : $INPUT"
    echo "    intrinsics : $INTRINSICS_JSON"
    echo "    out      : $OUT"

    EXTRA_ARGS=()
    [[ -n "$MAX_SECONDS" ]] && EXTRA_ARGS+=(--max_seconds "$MAX_SECONDS")

    # Run as the calling user so files written to the mounted output dir are
    # owned by them on the host, not by the image's user.
    docker run --rm --gpus all \
      --user "$(id -u):$(id -g)" \
      -e PYTHONUNBUFFERED=1 \
      ${HF_TOKEN_ARG:+-e HF_TOKEN="$HF_TOKEN_ARG"} \
      "${VOLUME_ARGS[@]}" \
      -v "$INPUT":"$C_INPUT":ro \
      -v "$OUT":"$C_OUTPUT" \
      "$IMAGE" \
      python inference/hand_detector_2d/scripts/run_hand_detector.py \
        --input "$C_INPUT" \
        --output_dir "$C_OUTPUT" \
        --bbox_detector_config "$C_CONFIG" \
        --intrinsics_json "$C_INTRINSICS_MOUNTED" \
        --hand "$HAND" \
        --device "$DEVICE" \
        --frame_batch_size "$FRAME_BATCH_SIZE" \
        "${EXTRA_ARGS[@]}"
    ;;

  *)
    usage
    ;;

esac

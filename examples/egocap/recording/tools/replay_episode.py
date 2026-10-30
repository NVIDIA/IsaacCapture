#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Real-data acceptance run: replay a recorded episode through RecordingService and verify the output.

This is a dataset reader plus an independent verifier, not an acquisition emulator: the service has no
replay mode. It reads an actual recorded episode (Zarr arrays + MP4 files), submits every selected row and
frame through the public API in deterministic round-robin order over the frozen source order (holding one
pending sample per source, retrying the same sample when the queue is FULL), then reopens the output on its
own and compares it with the input.

Usage (from any directory outside ``egocap`` itself)::

    <egocap>/.venv/bin/python recording/tools/replay_episode.py \
        --input EPISODE_DIR --manifest INPUT.sha256.json --output-root OUT_DIR [--rows N] [--report REPORT.json]
    # first time: add --write-manifest to create the SHA-256 manifest of the input files

The input is hashed before and after the run and must be unchanged. Exit status is 0 only if every check
passed; a missing input is reported as SKIPPED (exit 2), never as a pass.
"""

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import zarr

from egocap.recording.dataset_reader import EpisodeReader
from recording_service import (
    NumericSample,
    PixelFormat,
    QcConfig,
    RecordingService,
    ServiceConfig,
    SubmissionStatus,
    VideoSample,
)

#: Acceptance thresholds for the lossy re-encode (decoded output vs decoded input, 0-255 scale).
MAX_MEAN_ABS_ERROR = 4.0
MIN_PSNR_DB = 35.0
ENCODER_OPTIONS = {"crf": 18}
#: Frames sampled per camera for the identity check, and the margin by which a camera's own input must beat others.
IDENTITY_SAMPLES = 5
IDENTITY_MIN_RATIO = 1.5


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict[str, dict]:
    """Relative path -> {size, sha256} for every file under ``root``."""
    return {
        str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def verify_manifest(root: Path, manifest: dict[str, dict]) -> list[str]:
    """Differences between the files under ``root`` and the manifest (empty means identical)."""
    actual = build_manifest(root)
    problems = [f"missing: {p}" for p in manifest if p not in actual]
    problems += [f"unexpected: {p}" for p in actual if p not in manifest]
    problems += [f"changed: {p}" for p in manifest if p in actual and actual[p] != manifest[p]]
    return problems


def _errors(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(mean absolute error, PSNR dB) between two uint8 images, computed in floating point."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    mse = float(np.mean((x - y) ** 2))
    return float(np.mean(np.abs(x - y))), float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def sample_stream(reader: EpisodeReader, name: str, stop: int) -> Iterator:
    """Lazily yield service samples for rows/frames ``[0, stop)`` of one source (one decoded frame at a time)."""
    if reader.is_video(name):
        for f in reader.video_frames(name, 0, stop):
            yield VideoSample(name, f.image_rgb, PixelFormat.RGB8, f.capture_time, f.receive_time, f.intrinsics)
    else:
        for r in reader.numeric_rows(name, 0, stop):
            yield NumericSample(name, r.capture_time, r.receive_time, r.fields)


async def submit_round_robin(service: RecordingService, episode_id: str, streams: dict[str, Iterator]) -> int:
    """Submit the next sample of each non-exhausted source in frozen order, repeatedly; returns FULL retries.

    Only the sample currently being submitted is held, so memory stays bounded by the service queue.
    """
    retries, active = 0, dict(streams)
    while active:
        for name in list(active):
            sample = next(active[name], None)
            if sample is None:
                del active[name]
                continue
            while service.submit_sample(episode_id, sample).status is SubmissionStatus.FULL:
                retries += 1
                await asyncio.sleep(0.001)  # back-pressure wait: let the writer drain, then retry the same sample
    return retries


async def replay(reader: EpisodeReader, output_root: Path, rows: int | None) -> tuple[dict, object]:
    """Run the service over the selected rows; returns (selection description, EpisodeResult)."""
    names = reader.source_names()
    numeric = tuple(reader.numeric_spec(n) for n in names if not reader.is_video(n))
    video = tuple(
        type(spec)(**{**spec.__dict__, "encoder_options": ENCODER_OPTIONS})
        for spec in (reader.video_spec(n, "ffmpeg_h264") for n in names if reader.is_video(n))
    )
    config = ServiceConfig(
        output_root, numeric, video, QcConfig(required_sources=tuple(names)), queue_capacity=256
    )
    order = [s.name for s in numeric] + [s.name for s in video]
    stops = {n: reader.row_count(n) if rows is None else min(rows, reader.row_count(n)) for n in order}
    task = str(zarr.open_group(str(reader.path), mode="r").attrs["task_description"])
    service = RecordingService(config)
    try:
        await service.open_session()
        episode = await service.start_episode(task, "acceptance")
        retries = await submit_round_robin(
            service, episode.episode_id, {n: sample_stream(reader, n, stops[n]) for n in order}
        )
        result = await service.stop_episode(episode.episode_id)
    finally:
        closed = await service.close()
    info = {
        "order": order,
        "ranges": {n: [0, stops[n]] for n in order},
        "full_retries": retries,
        "close_released": closed.released,
    }
    return info, result


def _frame_at(reader: EpisodeReader, name: str, index: int) -> np.ndarray:
    """One decoded input frame (RGB)."""
    return next(reader.video_frames(name, index, index + 1)).image_rgb


def _output_frame_at(path: Path, index: int) -> np.ndarray:
    """One decoded output frame (RGB)."""
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"{path.name}: could not decode frame {index}")
    return np.ascontiguousarray(bgr[..., ::-1])


def identity_findings(reader: EpisodeReader, out_dir: Path, cameras: list[str], rows: dict[str, int]) -> dict:
    """Per camera: is the output closest to that camera's own input (not another camera's, nor a mirror)?

    For a few frames, the output frame's error against its own input frame must be below
    ``1/IDENTITY_MIN_RATIO`` of its error against every other camera's input frame at the same index and
    below its error against the mirrored own frame.
    """
    findings = {}
    for name in cameras:
        # Cross-camera comparison needs a frame index every camera has, so sample within the shortest stream.
        indices = np.linspace(0, min(rows[c] for c in cameras) - 1, IDENTITY_SAMPLES).astype(int)
        own, others, mirrored = [], {c: [] for c in cameras if c != name}, []
        for i in indices:
            out = _output_frame_at(out_dir / f"{name}.mp4", int(i))
            source = _frame_at(reader, name, int(i))
            own.append(_errors(out, source)[0])
            mirrored.append(_errors(out, source[:, ::-1])[0])
            for other in others:
                others[other].append(_errors(out, _frame_at(reader, other, int(i)))[0])
        own_mae = float(np.mean(own))
        other_mae = {c: float(np.mean(v)) for c, v in others.items()}
        mirror_mae = float(np.mean(mirrored))
        ok = all(own_mae * IDENTITY_MIN_RATIO < m for m in other_mae.values()) and own_mae < mirror_mae
        findings[name] = {
            "own_mae": own_mae,
            "other_cameras_mae": other_mae,
            "mirrored_mae": mirror_mae,
            "passed": ok,
        }
    return findings


def swapped_camera_control(reader: EpisodeReader, out_dir: Path, cameras: list[str], rows: dict[str, int]) -> dict:
    """Negative control: swap two cameras' MP4s in a COPY of the real output; the identity check must fail both."""
    if len(cameras) < 2:
        return {"skipped": "fewer than two cameras"}
    copy = out_dir.parent / (out_dir.name + ".swapped_control")
    shutil.copytree(out_dir, copy)
    try:
        first, second = cameras[0], cameras[1]
        (copy / f"{first}.mp4").rename(copy / "tmp.mp4")
        (copy / f"{second}.mp4").rename(copy / f"{first}.mp4")
        (copy / "tmp.mp4").rename(copy / f"{second}.mp4")
        found = identity_findings(reader, copy, [first, second], rows)
        return {"swapped": [first, second], "detected": not found[first]["passed"] and not found[second]["passed"]}
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def verify_output(reader: EpisodeReader, out_dir: Path, selection: dict) -> dict:
    """Independently reopen the output and compare it with the selected input; returns per-source findings."""
    src, out = zarr.open_group(str(reader.path), mode="r"), zarr.open_group(str(out_dir), mode="r")
    findings: dict[str, dict] = {}
    for name, (start, stop) in selection.items():
        found: dict = {"rows": stop - start, "problems": []}
        findings[name] = found
        for array_name, _ in src[name].arrays():
            a, b = src[name][array_name][start:stop], out[name][array_name][:]
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                found["problems"].append(f"{array_name}: values/dtype/shape differ")
        if not reader.is_video(name):
            continue
        cap = cv2.VideoCapture(str(out_dir / f"{name}.mp4"))
        count, worst_mae, worst_psnr, sum_mae = 0, 0.0, float("inf"), 0.0
        for frame_in in reader.video_frames(name, start, stop):
            ok, bgr = cap.read()
            if not ok:
                break
            frame_out = np.ascontiguousarray(bgr[..., ::-1])
            if frame_out.shape != frame_in.image_rgb.shape:
                found["problems"].append("frame dimensions differ")
                break
            mae, psnr = _errors(frame_out, frame_in.image_rgb)
            worst_mae, worst_psnr, sum_mae, count = (
                max(worst_mae, mae),
                min(worst_psnr, psnr),
                sum_mae + mae,
                count + 1,
            )
        extra, _ = cap.read()  # the output must end exactly where the selection ends
        cap.release()
        found.update(
            frames_compared=count, worst_mae=worst_mae, worst_psnr_db=worst_psnr, mean_mae=sum_mae / max(count, 1)
        )
        if count != stop - start:
            found["problems"].append(f"{count} frames decoded, expected {stop - start}")
        if extra:
            found["problems"].append("output video has frames beyond the selected range")
        if worst_mae > MAX_MEAN_ABS_ERROR or worst_psnr < MIN_PSNR_DB:
            found["problems"].append(
                f"video error above threshold (MAE<={MAX_MEAN_ABS_ERROR}, PSNR>={MIN_PSNR_DB})"
            )
    return findings


def main() -> int:
    """Parse arguments, run the acceptance replay, write the report, and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--rows", type=int, default=None, help="rows/frames per source (default: all of each source)"
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.rows is not None and args.rows <= 0:
        parser.error("--rows must be a positive integer")
    if not args.input.is_dir():
        print(f"SKIPPED: input episode {args.input} not found")
        return 2
    if args.write_manifest:
        args.manifest.write_text(json.dumps(build_manifest(args.input), indent=2, sort_keys=True))
    manifest = json.loads(args.manifest.read_text())
    problems = verify_manifest(args.input, manifest)
    if problems:
        print("input does not match its manifest:", *problems, sep="\n  ")
        return 1
    out_resolved, in_resolved = args.output_root.resolve(), args.input.resolve()
    if out_resolved == in_resolved or in_resolved in out_resolved.parents:
        print("output root must be separate from the input dataset")
        return 1

    reader = EpisodeReader(args.input)
    started = time.time()
    selection_info, result = asyncio.run(replay(reader, args.output_root, args.rows))
    elapsed = time.time() - started
    ranges = selection_info["ranges"]
    cameras = [n for n in selection_info["order"] if reader.is_video(n)]
    findings, identity, control = {}, {}, {}
    attrs: dict = {}
    if result.final_path:
        findings = verify_output(reader, result.final_path, ranges)
        identity = identity_findings(reader, result.final_path, cameras, {n: ranges[n][1] for n in cameras})
        control = swapped_camera_control(reader, result.final_path, cameras, {n: ranges[n][1] for n in cameras})
        attrs = dict(zarr.open_group(str(result.final_path), mode="r").attrs)
    after = verify_manifest(args.input, manifest)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent, check=False
    ).stdout.strip()
    checks = {
        "disposition_kept": result.disposition.value == "KEPT",
        "quality_passed": bool(result.quality and result.quality.outcome.value == "PASSED"),
        "service_released": selection_info["close_released"],
        "counts_accepted_equal_written": all(c.accepted == c.written for c in result.counts),
        "output_matches_input": bool(findings) and not any(f["problems"] for f in findings.values()),
        "camera_identity": bool(identity) and all(f["passed"] for f in identity.values()),
        "swapped_camera_control_detected": control.get("detected", "skipped" in control),
        "input_unchanged": not after,
        "metadata_present": bool(attrs.get("task_description")) and attrs.get("operator_name") == "acceptance",
        "no_false_domain_verdict": "freeze_check_passed" not in attrs and attrs.get("domain_checks") == "SKIPPED",
    }
    report = {
        "command": " ".join(sys.argv),
        "source_revision": commit,
        "input": str(args.input),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "encoder_options": attrs.get("video_encoder_options"),
        "selection": selection_info,
        "elapsed_seconds": round(elapsed, 2),
        "counts": {c.source_name: [c.accepted, c.written] for c in result.counts},
        "quality": [(r.name, r.outcome.value) for r in result.quality.check_results] if result.quality else None,
        "findings": findings,
        "camera_identity": identity,
        "swapped_camera_control": control,
        "input_changes_after": after,
        "checks": checks,
        "thresholds": {
            "max_mean_abs_error": MAX_MEAN_ABS_ERROR,
            "min_psnr_db": MIN_PSNR_DB,
            "identity_min_ratio": IDENTITY_MIN_RATIO,
        },
        "passed": all(checks.values()),
    }
    text = json.dumps(report, indent=2, default=str)
    if args.report:
        args.report.write_text(text)
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

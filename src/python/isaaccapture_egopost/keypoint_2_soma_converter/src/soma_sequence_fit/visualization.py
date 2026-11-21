# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Native skeleton overlays and input-to-fit video comparisons."""

from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

from .data import load_inputs, video_frames
from .geometry import FINGERS, project_np


INPUT_ROOT_COLOR = (255, 80, 255)


def draw_skeleton(canvas, points, K, color, parent_ids):
    """Draw fitted native SOMA joints using the model's parent hierarchy."""
    uv = project_np(points, K)
    usable = np.isfinite(uv).all(-1) & (np.abs(uv) < 1e6).all(-1)
    xy = np.zeros((len(points), 2), np.int32)
    xy[usable] = np.rint(uv[usable]).astype(np.int32)
    for child, parent in enumerate(parent_ids):
        if (
            0 <= parent < len(points)
            and parent != child
            and usable[parent]
            and usable[child]
        ):
            cv2.line(
                canvas,
                tuple(xy[parent]),
                tuple(xy[child]),
                (20, 20, 20),
                4,
                cv2.LINE_AA,
            )
            cv2.line(canvas, tuple(xy[parent]), tuple(xy[child]), color, 2, cv2.LINE_AA)
    h, w = canvas.shape[:2]
    for x, y in xy[usable]:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (int(x), int(y)), 4, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)


def draw_input_root(canvas, point, K, label):
    """Mark the original input joint 0 as a reference, without moving the fit."""
    uv = project_np(np.asarray(point)[None], K)[0]
    if not np.isfinite(uv).all():
        return
    h, w = canvas.shape[:2]
    if not (0 <= uv[0] < w and 0 <= uv[1] < h):
        return
    x, y = (int(v) for v in np.rint(uv))
    if not (0 <= x < w and 0 <= y < h):
        return
    cv2.circle(canvas, (x, y), 8, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 8, INPUT_ROOT_COLOR, 2, cv2.LINE_AA)
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    origin = (max(0, min(x + 12, w - text_width - 2)), max(14, min(y - 10, h - 4)))
    cv2.putText(
        canvas,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        INPUT_ROOT_COLOR,
        1,
        cv2.LINE_AA,
    )


def write_skeleton_overlay(data, joints25, valid, parent_ids, outdir):
    """Project all 25 native SOMA joints and bones onto the source video."""
    if joints25.shape != (data.nframes, 2, 25, 3) or parent_ids.shape != (2, 25):
        raise ValueError(
            "Overlay requires fitted joints25 and per-hand native parent indices."
        )
    writer = cv2.VideoWriter(
        str(Path(outdir) / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        data.fps,
        (data.width, data.height),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open MP4 writer; parameters were already saved.")
    colors = ((80, 210, 70), (60, 120, 240))
    try:
        for t, frame in tqdm(
            video_frames(data), total=data.nframes, desc="Skeleton overlay"
        ):
            canvas = frame.copy()
            for side, color in enumerate(colors):
                if valid[t, side]:
                    draw_skeleton(
                        canvas, joints25[t, side], data.K[t], color, parent_ids[side]
                    )
                draw_input_root(
                    canvas,
                    data.points[t, side, 0],
                    data.K[t],
                    ("L input root", "R input root")[side],
                )
            cv2.putText(
                canvas,
                f"frame {t} | fitted SOMA 25 joints",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            for side, label in enumerate(("left", "right")):
                cv2.putText(
                    canvas,
                    label,
                    (10 + side * 70, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colors[side],
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                canvas,
                "magenta ring: input root (joint 0)",
                (170, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                INPUT_ROOT_COLOR,
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()


def write_comparison(video, points_file, fit_file, output):
    """Write an H.264 comparison and last-frame JPEG from aligned input and fit data."""
    from imageio_ffmpeg import get_ffmpeg_exe

    output = Path(output)
    poster = output.with_suffix(".jpg")
    if output.exists() or poster.exists():
        raise ValueError("Comparison output already exists; choose a new output path.")
    data = load_inputs(video, points_file)
    with np.load(fit_file, allow_pickle=False) as fit:
        joints = np.array(fit["joints25"])
        valid = np.array(fit["valid"], dtype=bool)
        parents = np.array(fit["joint_parent_ids"])
        if (
            str(fit["units"].item()) != "m"
            or str(fit["coordinate_system"].item()) != "opencv_camera"
        ):
            raise ValueError("Fit must contain camera-space joints in meters.")
        if joints.shape != (data.nframes, 2, 25, 3) or valid.shape != (data.nframes, 2):
            raise ValueError(
                "Fit and input frame counts or hand dimensions do not match."
            )
        if parents.shape != (2, 25) or np.any((parents < -1) | (parents >= 25)):
            raise ValueError(
                "Fit must contain native parent indices for both 25-joint hands."
            )
        if not np.array_equal(fit["frame_index"], np.arange(data.nframes)):
            raise ValueError("Fit frame indices do not match the source video.")
        if not np.allclose(fit["K"], data.K) or not np.allclose(
            fit["timestamps"], data.timestamps
        ):
            raise ValueError(
                "Fit and input camera intrinsics or timestamps do not match."
            )

    input_parents = np.zeros(21, dtype=np.int32)
    for finger in FINGERS:
        input_parents[finger[1:]] = finger[:-1]
    colors = ((80, 210, 70), (60, 120, 240))
    # Keep the full image on both sides. H.264 yuv420p needs even dimensions.
    panel_width = data.width + data.width % 2
    height = data.height + data.height % 2
    width = 2 * panel_width
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(data.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-n",
        str(output),
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        try:
            for t, frame in tqdm(
                video_frames(data), total=data.nframes, desc="Comparison"
            ):
                panels = [frame.copy(), frame.copy()]
                for side, color in enumerate(colors):
                    draw_skeleton(
                        panels[0], data.points[t, side], data.K[t], color, input_parents
                    )
                    if valid[t, side]:
                        draw_skeleton(
                            panels[1], joints[t, side], data.K[t], color, parents[side]
                        )
                    draw_input_root(
                        panels[1],
                        data.points[t, side, 0],
                        data.K[t],
                        ("L input root", "R input root")[side],
                    )
                titles = (
                    "Input: 21 points (MANO convention)",
                    "Fitted: SOMA 25 joints",
                )
                for panel, title in zip(panels, titles):
                    cv2.rectangle(panel, (0, 0), (data.width, 64), (24, 24, 24), -1)
                    cv2.putText(
                        panel,
                        title,
                        (14, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        panel,
                        f"frame {t} | {t / data.fps:.2f}s",
                        (14, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )
                    for side, label in enumerate(("left hand", "right hand")):
                        cv2.putText(
                            panel,
                            label,
                            (280 + 150 * side, 52),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            colors[side],
                            1,
                            cv2.LINE_AA,
                        )
                cv2.putText(
                    panels[1],
                    "magenta ring: input root (joint 0)",
                    (600, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    INPUT_ROOT_COLOR,
                    1,
                    cv2.LINE_AA,
                )
                panels = [
                    cv2.copyMakeBorder(
                        p,
                        0,
                        height - data.height,
                        0,
                        panel_width - data.width,
                        cv2.BORDER_CONSTANT,
                    )
                    for p in panels
                ]
                canvas = np.concatenate(panels, axis=1)
                process.stdin.write(canvas.tobytes())
                if t == data.nframes - 1 and not cv2.imwrite(str(poster), canvas):
                    raise RuntimeError(f"Could not write comparison poster: {poster}")
            process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace"))
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            output.unlink(missing_ok=True)
            poster.unlink(missing_ok=True)
            raise
    print(f"Saved comparison: {output}")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stereo camera that is drawn instead of captured.

A pair of pinhole eyes turns slowly in a furnished room: a tiled floor, a
panelled ceiling with lamps, four walls of different colours with pictures
and a door, a rug, and a few boxes and pillars standing in it. Both eyes of a frame
are drawn for one instant, the calibration is exact and the camera's pose is
known, so a consumer that follows a camera has something to follow with no
hardware attached. Everything is drawn on the GPU, from this file alone: it
reads no image, model or other data.

The world's axes are the camera's own at rest: x right, y down, z forward.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

from ..pipeline import Frame, FrameSource, SourceSpec, StereoCalibration
from ._helpers import CaptureMailbox

#: The distance between the two eyes.
BASELINE_M = 0.063
#: The focal length in pixels, as a fraction of the image width: a
#: horizontal field of view of 77 degrees.
FOCAL_PER_WIDTH = 0.625
#: The room around the camera, which sits at the origin: its lowest and
#: highest x, y and z. The floor is 1.2 m under the eyes and the wall ahead
#: 5 m away, so at rest the floor, the ceiling and both side walls are in
#: view and their lines run toward that wall.
ROOM_MIN_M = (-2.5, -1.4, -3.0)
ROOM_MAX_M = (2.5, 1.2, 5.0)
#: What stands on the floor: centre, half extents and colour of each box.
#: The tall, thin ones are pillars that reach the ceiling.
_OBJECTS = (
    ((-1.1, 0.85, 2.4), (0.35, 0.35, 0.35), (204, 66, 54)),
    ((1.0, 0.95, 3.1), (0.5, 0.25, 0.35), (232, 190, 64)),
    ((-2.0, 0.75, 4.0), (0.3, 0.45, 0.3), (64, 112, 204)),
    ((1.9, -0.1, 4.2), (0.18, 1.3, 0.18), (196, 196, 192)),
    ((-1.9, -0.1, 1.2), (0.18, 1.3, 0.18), (196, 196, 192)),
)
#: The lamp that lights it all, under the ceiling ahead of the camera.
_LAMP_M = (0.0, -1.1, 2.0)
#: The camera holds still this long before it starts to turn.
STILL_S = 2.0

# How long stop() waits for the drawing thread.
_STOP_S = 5.0
#: It then swings left and right and nods, each as a sine.
YAW_DEG, YAW_PERIOD_S = 35.0, 10.0
PITCH_DEG, PITCH_PERIOD_S = 8.0, 7.0


def scene_rotation(seconds: float, still_s: float = STILL_S) -> np.ndarray:
    """The camera's orientation ``seconds`` after it started: the 3x3
    rotation that takes a vector from the camera's frame to the world's.
    Positive yaw turns it to the right, positive pitch tilts it up."""
    moving = max(0.0, seconds - still_s)
    yaw = math.radians(YAW_DEG) * math.sin(2.0 * math.pi * moving / YAW_PERIOD_S)
    pitch = math.radians(PITCH_DEG) * math.sin(2.0 * math.pi * moving / PITCH_PERIOD_S)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    about_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return about_y @ about_x


# One thread per pixel: its ray from the eye to the nearest surface, that
# surface's paint, and the lamp's light on it. Four rays per pixel soften
# the edges.
_DRAW_ROOM_KERNEL_SRC = r"""
// A number in 0..1 that looks unrelated between neighbouring cells.
static __device__ float speckle(int i, int j, int salt) {
    unsigned int a = (unsigned int)i * 1000003u ^ (unsigned int)j * 15485863u
        ^ (unsigned int)salt * 998244353u;
    for (int round = 0; round < 3; ++round) { a *= 2654435761u; a ^= a >> 15; }
    return (a & 1023u) / 1023.0f;
}

static __device__ float ease(float x) {
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

static __device__ float cell(float x, float size, int* index) {
    const float scaled = x / size;
    *index = (int)floorf(scaled);
    return scaled - *index;  // where in the cell, 0..1
}

static __device__ void set(float* c, float r, float g, float b) {
    c[0] = r; c[1] = g; c[2] = b;
}

static __device__ void scale(float* c, float k) {
    c[0] *= k; c[1] *= k; c[2] *= k;
}

// Checkered tiles with dark joints, and a rug ahead of the camera.
static __device__ void paint_floor(float x, float z, float* c) {
    int i, j;
    if (fabsf(x) < 0.9f && z > 2.0f && z < 4.2f) {
        if (fminf(0.9f - fabsf(x), fminf(z - 2.0f, 4.2f - z)) < 0.08f) {
            set(c, 96.0f, 30.0f, 36.0f);
            return;
        }
        cell(x + z, 0.21f, &i); cell(x - z, 0.21f, &j);  // diamonds
        if ((i + j) & 1) set(c, 150.0f, 44.0f, 52.0f); else set(c, 226.0f, 198.0f, 150.0f);
        scale(c, 0.86f + 0.14f * speckle(i, j, 4));
        return;
    }
    const float fx = cell(x, 0.5f, &i), fz = cell(z, 0.5f, &j);
    if (fx < 0.03f || fz < 0.03f) { set(c, 60.0f, 48.0f, 38.0f); return; }
    if ((i + j) & 1) set(c, 168.0f, 132.0f, 96.0f); else set(c, 120.0f, 92.0f, 66.0f);
    scale(c, 0.9f + 0.2f * speckle(i, j, 3));
}

// Light panels, and a lamp in every second one each way. Returns whether
// the point is on a lamp, which shines by itself.
static __device__ bool paint_ceiling(float x, float z, float* c) {
    int i, j;
    const float fx = cell(x, 1.0f, &i), fz = cell(z, 1.0f, &j);
    if (!(i & 1) && !(j & 1) && fabsf(fx - 0.5f) < 0.22f && fabsf(fz - 0.5f) < 0.22f) {
        set(c, 255.0f, 250.0f, 235.0f);
        return true;
    }
    if (fx < 0.03f || fz < 0.03f) {
        set(c, 150.0f, 150.0f, 146.0f);
    } else if ((i & 1) && (j & 1) && fabsf(fx - 0.5f) < 0.12f && fabsf(fz - 0.5f) < 0.12f) {
        set(c, 112.0f, 112.0f, 110.0f);  // a vent
    } else {
        set(c, 214.0f, 214.0f, 206.0f);
        scale(c, 0.9f + 0.1f * speckle(i, j, 5));
    }
    return false;
}

// A wall: ``along`` runs along it, ``up`` is the height above the floor.
// From the floor up: a skirting board, panelling, a rail, bricks, and a
// frieze of small tiles under the ceiling, with a picture on three of the
// walls and a door on the fourth.
static __device__ void paint_wall(int wall, float along, float up, float* c) {
    const float upper[4][3] = {{112, 150, 196}, {160, 140, 186}, {120, 176, 150}, {200, 140, 104}};
    const float lower[4][3] = {{70, 100, 140}, {112, 96, 136}, {78, 124, 104}, {150, 96, 70}};
    if (up < 0.10f) { set(c, 50.0f, 46.0f, 44.0f); return; }
    if (wall == 1 && fabsf(along) < 0.45f && up < 2.0f) {  // the door
        const bool panel = fabsf(along) < 0.33f && (fabsf(up - 0.55f) < 0.33f || fabsf(up - 1.45f) < 0.40f);
        if (panel) set(c, 120.0f, 84.0f, 56.0f); else set(c, 92.0f, 62.0f, 40.0f);
        return;
    }
    const float centre = wall < 2 ? 0.0f : 2.6f, half = wall < 2 ? 0.8f : 0.6f;
    if (wall != 1 && fabsf(along - centre) < half && up > 1.25f && up < 2.05f) {  // a picture
        const float border = fminf(fminf(along - centre + half, centre + half - along),
                                   fminf(up - 1.25f, 2.05f - up));
        if (border < 0.05f) { set(c, 40.0f, 34.0f, 30.0f); return; }
        int i, j;
        cell(along, 0.1f, &i); cell(up, 0.1f, &j);
        set(c, 50.0f + 190.0f * speckle(i, j, 20 + wall), 50.0f + 190.0f * speckle(i, j, 30 + wall),
            50.0f + 190.0f * speckle(i, j, 40 + wall));
        return;
    }
    int i, j;
    if (up > 2.25f && up < 2.41f) {  // the frieze
        cell(along, 0.08f, &i); cell(up, 0.08f, &j);
        set(c, lower[wall][0], lower[wall][1], lower[wall][2]);
        scale(c, speckle(i, j, 60 + wall) > 0.5f ? 1.5f : 0.7f);
        return;
    }
    if (up < 0.90f) {  // panelling, in upright boards
        const float f = cell(along, 0.25f, &i);
        set(c, lower[wall][0], lower[wall][1], lower[wall][2]);
        scale(c, (0.88f + 0.24f * speckle(i, wall, 7)) * (f < 0.04f || up > 0.86f ? 0.72f : 1.0f));
        return;
    }
    const float fu = cell(up, 0.25f, &j);
    const float fa = cell(along + 0.25f * (j & 1), 0.5f, &i);
    set(c, upper[wall][0], upper[wall][1], upper[wall][2]);
    scale(c, (0.8f + 0.4f * speckle(i, j, 11 + wall)) * (fu < 0.06f || fa < 0.03f ? 0.74f : 1.0f));
}

extern "C" __global__ void draw_room(
        const float* __restrict__ eye,      // rotation world<-camera, row-major, then position
        const float* __restrict__ room,     // lowest x y z, then highest
        const float* __restrict__ objects,  // per object: centre, half extents, colour
        const int count,                    // how many objects
        const float* __restrict__ lamp,
        const float fx, const float fy, const float cx, const float cy,
        const int W, const int H,
        unsigned char* __restrict__ out) {  // (H, W, 4) RGBA
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const float floor_y = room[4];
    float sum[3] = {0.0f, 0.0f, 0.0f};
    for (int s = 0; s < 4; ++s) {
        const float px = x + 0.25f + 0.5f * (s & 1), py = y + 0.25f + 0.5f * (s >> 1);
        const float c[3] = {(px - 0.5f - cx) / fx, (py - 0.5f - cy) / fy, 1.0f};
        float d[3], o[3];
        for (int k = 0; k < 3; ++k) {
            d[k] = eye[3 * k] * c[0] + eye[3 * k + 1] * c[1] + eye[3 * k + 2] * c[2];
            o[k] = eye[9 + k];
        }

        // The eye is inside the room: the wall the ray leaves through.
        float t = 1e30f;
        int axis = 0, object = -1;
        for (int k = 0; k < 3; ++k) {
            if (fabsf(d[k]) < 1e-9f) continue;
            const float tk = ((d[k] > 0.0f ? room[3 + k] : room[k]) - o[k]) / d[k];
            if (tk < t) { t = tk; axis = k; }
        }
        // Unless a box stands in the way: where the ray enters it.
        for (int n = 0; n < count; ++n) {
            const float* box = objects + 9 * n;
            float enter = 0.0f, leave = t;
            int entered = 0;
            for (int k = 0; k < 3; ++k) {
                if (fabsf(d[k]) < 1e-9f) {
                    if (fabsf(o[k] - box[k]) > box[3 + k]) { leave = -1.0f; break; }
                    continue;
                }
                const float a = (box[k] - box[3 + k] - o[k]) / d[k];
                const float b = (box[k] + box[3 + k] - o[k]) / d[k];
                const float lo = fminf(a, b), hi = fmaxf(a, b);
                if (lo > enter) { enter = lo; entered = k; }
                leave = fminf(leave, hi);
            }
            if (enter > 0.0f && enter < leave) { t = enter; axis = entered; object = n; }
        }

        const float p[3] = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
        float normal[3] = {0.0f, 0.0f, 0.0f};
        normal[axis] = d[axis] > 0.0f ? -1.0f : 1.0f;  // facing the eye
        float rgb[3];
        float glow = 0.0f, shade = 1.0f;
        if (object >= 0) {
            const float* box = objects + 9 * object;
            set(rgb, box[6], box[7], box[8]);
            const float u = p[(axis + 1) % 3] - box[(axis + 1) % 3];
            const float v = p[(axis + 2) % 3] - box[(axis + 2) % 3];
            const float hu = box[3 + (axis + 1) % 3], hv = box[3 + (axis + 2) % 3];
            int i, j;
            cell(u, 0.175f, &i); cell(v, 0.175f, &j);
            shade = 0.72f + 0.28f * speckle(i, j, 50 + 3 * object + axis);
            if (box[4] > 1.0f) {  // a pillar: bands up its height
                cell(p[1], 0.4f, &j);
                if (j & 1) shade *= 0.8f;
            }
            if (fminf(hu - fabsf(u), hv - fabsf(v)) < 0.02f) shade *= 0.6f;  // its edges
        } else if (axis == 1 && d[1] > 0.0f) {
            paint_floor(p[0], p[2], rgb);
            for (int n = 0; n < count; ++n) {  // darker where a box stands
                const float* box = objects + 9 * n;
                const float gx = fmaxf(fabsf(p[0] - box[0]) - box[3], 0.0f);
                const float gz = fmaxf(fabsf(p[2] - box[2]) - box[5], 0.0f);
                shade = fminf(shade, 0.55f + 0.45f * ease(sqrtf(gx * gx + gz * gz) / 0.3f));
            }
        } else if (axis == 1) {
            if (paint_ceiling(p[0], p[2], rgb)) glow = 1.0f;
        } else if (axis == 2) {
            paint_wall(d[2] > 0.0f ? 0 : 1, p[0], floor_y - p[1], rgb);
        } else {
            paint_wall(d[0] > 0.0f ? 3 : 2, p[2], floor_y - p[1], rgb);
        }
        if (object < 0) {
            // Darker toward the edges of a wall, where it meets the next one.
            const int a = (axis + 1) % 3, b = (axis + 2) % 3;
            const float edge = fminf(fminf(p[a] - room[a], room[3 + a] - p[a]),
                                     fminf(p[b] - room[b], room[3 + b] - p[b]));
            shade *= 0.68f + 0.32f * ease(edge / 0.5f);
        }
        const float l[3] = {lamp[0] - p[0], lamp[1] - p[1], lamp[2] - p[2]};
        const float far2 = l[0] * l[0] + l[1] * l[1] + l[2] * l[2];
        const float facing = fmaxf(
            (normal[0] * l[0] + normal[1] * l[1] + normal[2] * l[2]) * rsqrtf(far2 + 1e-9f), 0.0f);
        const float light = glow > 0.0f ? 1.0f
            : shade * (0.52f + 0.62f * facing / (1.0f + 0.03f * far2));
        sum[0] += fminf(rgb[0] * light, 255.0f);
        sum[1] += fminf(rgb[1] * light, 255.0f);
        sum[2] += fminf(rgb[2] * light, 255.0f);
    }
    unsigned char* pixel = out + 4 * ((long long)y * W + x);
    pixel[0] = (unsigned char)(0.25f * sum[0] + 0.5f);
    pixel[1] = (unsigned char)(0.25f * sum[1] + 0.5f);
    pixel[2] = (unsigned char)(0.25f * sum[2] + 0.5f);
    pixel[3] = 255;
}
"""


class SyntheticSceneSource(FrameSource):
    """Both eyes of each drawing of the room, in one frame.

    A frame is stamped when it is drawn, and ``rotation_at`` gives the
    camera's true orientation for that stamp.
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        fps: float = 30.0,
        still_s: float = STILL_S,
    ) -> None:
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError(
                "SyntheticSceneSource requires CuPy. Run `camera_viz.sh setup` "
                "to install the package matching your CUDA toolkit."
            ) from e
        if width <= 0 or height <= 0:
            raise ValueError(f"SyntheticSceneSource: bad size {width}x{height}")
        self._cp = cp
        self._spec = SourceSpec(
            name=name, width=width, height=height, pixel_format="rgba8"
        )
        self._interval_s = 1.0 / fps if fps > 0.0 else 0.0
        self._still_s = float(still_s)
        focal = FOCAL_PER_WIDTH * width
        self._intrinsics = (focal, focal, (width - 1) / 2.0, (height - 1) / 2.0)

        # Not zeroed: a slot is drawn in full before it is published.
        self._left = [cp.empty((height, width, 4), dtype=cp.uint8) for _ in range(3)]
        self._right = [cp.empty((height, width, 4), dtype=cp.uint8) for _ in range(3)]
        self._captures = CaptureMailbox(len(self._left))
        self._kernel = cp.RawKernel(_DRAW_ROOM_KERNEL_SRC, "draw_room")
        # A stream of its own, so drawing does not wait for the reader's
        # work on the default stream.
        self._stream = cp.cuda.Stream(non_blocking=True)
        self._lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0_ns = 0

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    @property
    def dropped_captures(self) -> int:
        """Drawings that were replaced before they were read."""
        return self._captures.dropped

    def calibration(self) -> StereoCalibration:
        fx, fy, cx, cy = self._intrinsics
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return StereoCalibration(
            width=self._spec.width,
            height=self._spec.height,
            left_K=K,
            left_D=np.zeros(5),
            right_K=K.copy(),
            right_D=np.zeros(5),
            R=np.eye(3),
            T=np.array([-BASELINE_M, 0.0, 0.0]),
            rectified=True,
        )

    def rotation_at(self, timestamp_ns: int) -> np.ndarray:
        """The camera's true orientation at the timestamp of a frame of
        the current session, as ``scene_rotation`` gives it."""
        return scene_rotation((timestamp_ns - self._t0_ns) * 1e-9, self._still_s)

    def start(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError(f"Source {self._spec.name!r} is still stopping")
                return
            # stop() gave up waiting for this thread; it has ended since.
            self._thread = None
        self._stop.clear()
        with self._lock:
            self._error = None
        # Nothing drawn for the last session is handed out in this one.
        self._captures.reset()
        self._t0_ns = time.monotonic_ns()
        self._thread = threading.Thread(
            target=self._produce_loop, name=f"scene_{self._spec.name}", daemon=False
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_S)
            if self._thread.is_alive():
                return  # still compiling or drawing; it ends on its own
            self._thread = None

    def latest(self) -> Optional[Frame]:
        with self._lock:
            if self._error is not None:
                raise RuntimeError(
                    f"Source {self._spec.name!r} failed: {self._error}. "
                    "Run camera_viz.sh setup to check the CUDA environment."
                ) from self._error
        taken = self._captures.take()
        if taken is None:
            return None
        index, timestamp_ns = taken
        return Frame(
            image=self._left[index],
            image_right=self._right[index],
            timestamp_ns=timestamp_ns,
            source_id=self._spec.name,
            stream=0,
        )

    def _produce_loop(self) -> None:
        try:
            self._draw_frames()
        except BaseException as exc:
            # latest() forwards the failure to the reader.
            with self._lock:
                self._error = exc

    def _draw_frames(self) -> None:
        cp = self._cp
        width, height = self._spec.width, self._spec.height
        fx, fy, cx, cy = (np.float32(v) for v in self._intrinsics)
        grid, block = ((width + 15) // 16, (height + 15) // 16, 1), (16, 16, 1)
        # Pinned to the GPU the buffers are on, which need not be this
        # thread's default.
        with cp.cuda.Device(int(self._left[0].device.id)):
            # On the stream that draws, so the drawing waits for them.
            with self._stream:
                room = cp.asarray(ROOM_MIN_M + ROOM_MAX_M, dtype=cp.float32)
                objects = cp.asarray(
                    [value for box in _OBJECTS for part in box for value in part],
                    dtype=cp.float32,
                )
                lamp = cp.asarray(_LAMP_M, dtype=cp.float32)
            while not self._stop.is_set():
                drawn_at = time.monotonic()
                timestamp_ns = time.monotonic_ns()
                rotation = self.rotation_at(timestamp_ns)
                index = self._captures.write_index()
                with self._stream:
                    for buffers, side in ((self._left, -0.5), (self._right, 0.5)):
                        position = rotation @ np.array([side * BASELINE_M, 0.0, 0.0])
                        eye = cp.asarray(
                            np.concatenate([rotation.ravel(), position]),
                            dtype=cp.float32,
                        )
                        self._kernel(
                            grid,
                            block,
                            (
                                eye,
                                room,
                                objects,
                                np.int32(len(_OBJECTS)),
                                lamp,
                                fx,
                                fy,
                                cx,
                                cy,
                                np.int32(width),
                                np.int32(height),
                                buffers[index],
                            ),
                        )
                # The reader's copy runs on another stream.
                self._stream.synchronize()
                self._captures.publish(index, timestamp_ns)
                if self._interval_s > 0.0:
                    remaining = self._interval_s - (time.monotonic() - drawn_at)
                    if remaining > 0.0:
                        self._stop.wait(min(remaining, threading.TIMEOUT_MAX))

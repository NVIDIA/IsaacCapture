# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller guide, composited into the twin's stereo RGBD."""

from importlib.resources import files
import math
import time

import numpy as np

# Televiz currently requires ProjectionLayer to be the session's only layer.
# Composite panel depth too, so CloudXR reprojects the guide at its own distance.
# TODO(1.7): Use a native quad layer once Televiz supports and tests combining
# projection and quad layers.
_OVERLAY = r"""
extern "C" __global__ void overlay(
    unsigned char* color, float* depth, const unsigned char* image,
    int width, int height, int image_width, int image_height,
    const float* eye_to_anchor, float left, float right, float up, float down,
    float panel_width, float panel_height, float distance, float offset_y, float pitch,
    float near_z, float far_z)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= width * height) return;
    float rx = left + ((i % width) + 0.5f) / width * (right - left);
    float ry = up + ((i / width) + 0.5f) / height * (down - up);
    float dx = eye_to_anchor[0]*rx + eye_to_anchor[1]*ry - eye_to_anchor[2];
    float dy = eye_to_anchor[4]*rx + eye_to_anchor[5]*ry - eye_to_anchor[6];
    float dz = eye_to_anchor[8]*rx + eye_to_anchor[9]*ry - eye_to_anchor[10];
    float c = cosf(pitch), s = sinf(pitch);
    float denominator = -s*dy + c*dz;
    if (denominator >= -1e-6f) return;
    float oy = eye_to_anchor[7] - offset_y;
    float oz = eye_to_anchor[11] + distance;
    float t = (s*oy - c*oz) / denominator;
    if (t <= near_z || t >= far_z) return;
    float u = (eye_to_anchor[3] + t*dx) / panel_width + 0.5f;
    float v = 0.5f - (c*(oy + t*dy) + s*(oz + t*dz)) / panel_height;
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return;
    float sx = fminf(fmaxf(u * image_width - 0.5f, 0), image_width - 1);
    float sy = fminf(fmaxf(v * image_height - 0.5f, 0), image_height - 1);
    int x0 = (int)sx, y0 = (int)sy;
    int x1 = min(x0 + 1, image_width - 1);
    int y1 = min(y0 + 1, image_height - 1);
    float fx = sx - x0, fy = sy - y0;
    for (int c = 0; c < 3; ++c) {
        float top = (1-fx)*image[4*(y0*image_width+x0)+c]
                         + fx*image[4*(y0*image_width+x1)+c];
        float bottom = (1-fx)*image[4*(y1*image_width+x0)+c]
                            + fx*image[4*(y1*image_width+x1)+c];
        color[4*i+c] = (unsigned char)((1-fy)*top + fy*bottom + 0.5f);
    }
    color[4*i+3] = 255;
    depth[i] = far_z / (far_z-near_z) * (1-near_z/t);
}
"""


class _LazyPlacement:
    """Hold a world anchor, then ease to a new one after sustained head drift."""

    def __init__(self):
        self.position = None
        self.yaw = 0.0
        self._away_since = None
        self._transition = None

    def update(self, position, rotation, now):
        forward = -rotation[:, 2]
        yaw = (
            math.atan2(-forward[0], -forward[2])
            if np.linalg.norm(forward[[0, 2]]) > 1e-6
            else self.yaw
        )
        if self.position is None:
            self.position, self.yaw = position.copy(), yaw

        # Keep the anchor yaw-only so looking up does not move or tilt the guide.
        if self._transition is None:
            angle = math.atan2(math.sin(yaw - self.yaw), math.cos(yaw - self.yaw))
            away = (
                abs(angle) > math.radians(30)
                or np.linalg.norm(position - self.position) > 0.35
            )
            if not away:
                self._away_since = None
            elif self._away_since is None:
                self._away_since = now
            elif now - self._away_since >= 0.5:
                self._transition = (
                    now,
                    self.position.copy(),
                    self.yaw,
                    position.copy(),
                    angle,
                )

        if self._transition is not None:
            start, origin, start_yaw, target, angle = self._transition
            t = min((now - start) / 0.45, 1.0)
            blend = t * t * (3.0 - 2.0 * t)
            self.position = origin + blend * (target - origin)
            self.yaw = start_yaw + blend * angle
            if t >= 1.0:
                self._transition = None
                self._away_since = None

        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.position, np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


class ControllerGuide:
    """An opaque, lazily following plane drawn over the scene on the render thread."""

    def __init__(self, near_z, far_z):
        import cupy as cp
        from PIL import Image

        self._images = []
        for state in ("disengaged", "engaged"):
            asset = files(__package__).joinpath(f"controller-guide-{state}.png")
            with asset.open("rb") as source, Image.open(source) as image:
                self._images.append(cp.asarray(np.array(image.convert("RGBA"))))
        self._kernel = cp.RawKernel(_OVERLAY, "overlay")
        self._near_z, self._far_z = near_z, far_z
        self._placement = _LazyPlacement()
        self.colors = []
        self.depths = []

    def draw(self, twin, poses, fovs, engaged=False):
        import cupy as cp

        from isaaccapture.viz.robot.quaternion import to_matrix

        poses = np.asarray(poses).reshape(-1, 7)
        fovs = np.asarray(fovs).reshape(-1, 4)
        valid = np.all(np.isfinite(poses)) and np.allclose(
            np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-3
        )
        if valid:
            anchor_position, anchor_rotation = self._placement.update(
                poses[0, :3], to_matrix(poses[0, 3:]), time.monotonic()
            )
        pixels = self._images[bool(engaged)]
        image_height, image_width = pixels.shape[:2]
        for eye, (pose, fov) in enumerate(zip(poses, fovs)):
            source_color = cp.asarray(twin.color(eye))
            source_depth = cp.asarray(twin.depth(eye))
            # GL readback buffers are registered CUDA read-only, despite their
            # array interface reporting writable. Overlay into owned copies.
            if eye == len(self.colors):
                self.colors.append(cp.empty_like(source_color))
                self.depths.append(cp.empty_like(source_depth))
            color, depth = self.colors[eye], self.depths[eye]
            cp.copyto(color, source_color)
            cp.copyto(depth, source_depth)
            if not valid:
                continue
            transform = np.empty((3, 4), dtype=np.float32)
            transform[:, :3] = anchor_rotation.T @ to_matrix(pose[3:])
            transform[:, 3] = anchor_rotation.T @ (pose[:3] - anchor_position)
            height, width = depth.shape
            self._kernel(
                ((width * height + 255) // 256,),
                (256,),
                (
                    color,
                    depth,
                    pixels,
                    np.int32(width),
                    np.int32(height),
                    np.int32(image_width),
                    np.int32(image_height),
                    cp.asarray(transform),
                    *(np.float32(x) for x in np.tan(fov)),
                    np.float32(1.4),
                    np.float32(1.4 * image_height / image_width),
                    np.float32(1.1),
                    # Keep the entire panel above the forward eyeline.
                    np.float32(0.65),
                    # Fixed downward tilt faces the anchor without tracking head pitch.
                    np.float32(math.atan2(0.65, 1.1)),
                    np.float32(self._near_z),
                    np.float32(self._far_z),
                ),
            )
        # Finish both the readback copies and overlay before native submission
        # or the next render hands source-buffer ownership back to OpenGL.
        cp.cuda.get_current_stream().synchronize()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Differentiable silhouettes: CUDA rasterization or a portable soft reference."""

import torch
from .geometry import project


class SilhouetteRenderer:
    def __init__(self, backend="auto", device="cpu", size=128):
        self.size = size
        self.backend = backend
        if backend not in {"auto", "soft", "triton"}:
            raise ValueError(f"Unknown renderer backend: {backend}")
        if backend == "auto":
            self.backend = "soft"
            if str(device).startswith("cuda"):
                try:
                    from .fast_render import soft_union
                except ImportError:
                    pass
                else:
                    self.backend = "triton"
        if self.backend == "triton":
            if not str(device).startswith("cuda"):
                raise ValueError("The fused renderer requires CUDA.")
            from .fast_render import soft_union

            self.soft_union = soft_union
        if self.backend == "soft" and size > 128:
            raise ValueError(
                "The portable renderer is expensive; use size <=128 or the Triton backend on CUDA."
            )

    def __call__(self, vertices, faces, K, sigma=0.3):
        if self.backend == "triton":
            return self.soft_union(
                project(vertices, K),
                faces,
                (vertices[:, faces, 2] > 0.001).all(-1),
                self.size,
                sigma,
            )
        return self.soft(vertices, faces, K, sigma)

    def soft(self, vertices, faces, K, sigma):
        """Batched soft triangle union; portable reference implementation."""
        uv = project(vertices, K)
        yy, xx = torch.meshgrid(
            torch.arange(self.size, device=uv.device, dtype=uv.dtype),
            torch.arange(self.size, device=uv.device, dtype=uv.dtype),
            indexing="ij",
        )
        pixels = torch.stack((xx, yy), -1).reshape(1, 1, -1, 2)
        triangles = uv[:, faces]
        valid_z = (vertices[:, faces, 2] > 0.001).all(-1)
        accum = torch.zeros(
            len(vertices), self.size * self.size, device=uv.device, dtype=uv.dtype
        )
        for start in range(0, len(faces), 64):
            t = triangles[:, start : start + 64]
            a, end = t, t.roll(-1, dims=2)
            e = end - a
            area = e[:, :, 0, 0] * (-e[:, :, 2, 1]) - e[:, :, 0, 1] * (-e[:, :, 2, 0])
            sign = torch.where(area >= 0, 1.0, -1.0)
            distances = []
            for j in range(3):
                delta = pixels - a[:, :, j, None, :]
                cross = (
                    e[:, :, j, 0, None] * delta[..., 1]
                    - e[:, :, j, 1, None] * delta[..., 0]
                )
                distances.append(
                    cross
                    * sign[:, :, None]
                    / torch.linalg.vector_norm(e[:, :, j], dim=-1).clamp_min(1e-6)[
                        :, :, None
                    ]
                )
            signed = torch.stack(distances, -1).amin(-1)
            coverage = torch.sigmoid(signed / sigma)
            valid = valid_z[:, start : start + 64] & (area.abs() > 1e-8)
            accum = torch.maximum(accum, (coverage * valid[:, :, None]).amax(1))
        # Max-union prevents hundreds of off-triangle tails from inflating the silhouette.
        return accum.clamp(0, 1).reshape(-1, self.size, self.size)

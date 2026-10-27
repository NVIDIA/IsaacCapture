# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Fused CUDA soft triangle union with an analytic first-order backward.

Matches the portable renderer's max-union model without materializing the
batch x faces x pixels x edges autograd graph. Requires optional Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _union_forward(
    UV,
    FACES,
    VALID,
    OUT,
    WINNER,
    NV: tl.constexpr,
    NF: tl.constexpr,
    SIZE: tl.constexpr,
    SIGMA: tl.constexpr,
    BP: tl.constexpr,
    BF: tl.constexpr,
):
    batch = tl.program_id(0)
    pixels = tl.program_id(1) * BP + tl.arange(0, BP)
    px = (pixels % SIZE).to(tl.float32)[:, None]
    py = (pixels // SIZE).to(tl.float32)[:, None]
    best = tl.full((BP,), -float("inf"), tl.float32)
    winner = tl.full((BP,), -1, tl.int32)
    for start in range(tl.cdiv(NF, BF)):
        faces = start * BF + tl.arange(0, BF)
        mask = faces < NF
        ia = tl.load(FACES + faces * 3, mask, 0)
        ib = tl.load(FACES + faces * 3 + 1, mask, 0)
        ic = tl.load(FACES + faces * 3 + 2, mask, 0)
        ax = tl.load(UV + (batch * NV + ia) * 2)[None, :]
        ay = tl.load(UV + (batch * NV + ia) * 2 + 1)[None, :]
        bx = tl.load(UV + (batch * NV + ib) * 2)[None, :]
        by = tl.load(UV + (batch * NV + ib) * 2 + 1)[None, :]
        cx = tl.load(UV + (batch * NV + ic) * 2)[None, :]
        cy = tl.load(UV + (batch * NV + ic) * 2 + 1)[None, :]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        sign = tl.where(area >= 0, 1.0, -1.0)
        d0 = (
            sign
            * ((bx - ax) * (py - ay) - (by - ay) * (px - ax))
            / tl.maximum(
                tl.sqrt(((bx - ax) * (bx - ax)) + ((by - ay) * (by - ay))), 1.0e-6
            )
        )
        d1 = (
            sign
            * ((cx - bx) * (py - by) - (cy - by) * (px - bx))
            / tl.maximum(
                tl.sqrt(((cx - bx) * (cx - bx)) + ((cy - by) * (cy - by))), 1.0e-6
            )
        )
        d2 = (
            sign
            * ((ax - cx) * (py - cy) - (ay - cy) * (px - cx))
            / tl.maximum(
                tl.sqrt(((ax - cx) * (ax - cx)) + ((ay - cy) * (ay - cy))), 1.0e-6
            )
        )
        valid = mask & tl.load(VALID + batch * NF + faces, mask, 0)
        signed = tl.where(
            valid[None, :] & (tl.abs(area) > 1.0e-8),
            tl.minimum(tl.minimum(d0, d1), d2),
            -float("inf"),
        )
        local = tl.max(signed, 1)
        local_face = tl.min(
            tl.where(signed == local[:, None], faces[None, :], 2147483647), 1
        )
        update = local > best
        winner = tl.where(update, local_face, winner)
        best = tl.maximum(best, local)
    coverage = 1.0 / (1.0 + tl.exp(-best / SIGMA))
    index = batch * SIZE * SIZE + pixels
    tl.store(OUT + index, coverage, pixels < SIZE * SIZE)
    tl.store(WINNER + index, winner, pixels < SIZE * SIZE)


@triton.jit
def _union_backward(
    UV,
    FACES,
    OUT,
    WINNER,
    GRAD,
    DUV,
    NV: tl.constexpr,
    SIZE: tl.constexpr,
    SIGMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    pixels = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = batch * SIZE * SIZE + pixels
    face = tl.load(WINNER + offset, pixels < SIZE * SIZE, -1)
    valid = (pixels < SIZE * SIZE) & (face >= 0)
    ia = tl.load(FACES + face * 3, valid, 0)
    ib = tl.load(FACES + face * 3 + 1, valid, 0)
    ic = tl.load(FACES + face * 3 + 2, valid, 0)
    ax = tl.load(UV + (batch * NV + ia) * 2)
    ay = tl.load(UV + (batch * NV + ia) * 2 + 1)
    bx = tl.load(UV + (batch * NV + ib) * 2)
    by = tl.load(UV + (batch * NV + ib) * 2 + 1)
    cx = tl.load(UV + (batch * NV + ic) * 2)
    cy = tl.load(UV + (batch * NV + ic) * 2 + 1)
    px = (pixels % SIZE).to(tl.float32)
    py = (pixels // SIZE).to(tl.float32)
    sign = tl.where((bx - ax) * (cy - ay) - (by - ay) * (cx - ax) >= 0, 1.0, -1.0)
    l0 = tl.maximum(tl.sqrt(((bx - ax) * (bx - ax)) + ((by - ay) * (by - ay))), 1.0e-6)
    l1 = tl.maximum(tl.sqrt(((cx - bx) * (cx - bx)) + ((cy - by) * (cy - by))), 1.0e-6)
    l2 = tl.maximum(tl.sqrt(((ax - cx) * (ax - cx)) + ((ay - cy) * (ay - cy))), 1.0e-6)
    d0 = sign * ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / l0
    d1 = sign * ((cx - bx) * (py - by) - (cy - by) * (px - bx)) / l1
    d2 = sign * ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) / l2
    edge = tl.where((d0 <= d1) & (d0 <= d2), 0, tl.where(d1 <= d2, 1, 2))
    start = tl.where(edge == 0, ia, tl.where(edge == 1, ib, ic))
    end = tl.where(edge == 0, ib, tl.where(edge == 1, ic, ia))
    sx = tl.where(edge == 0, ax, tl.where(edge == 1, bx, cx))
    sy = tl.where(edge == 0, ay, tl.where(edge == 1, by, cy))
    ex = tl.where(edge == 0, bx, tl.where(edge == 1, cx, ax))
    ey = tl.where(edge == 0, by, tl.where(edge == 1, cy, ay))
    dx, dy = ex - sx, ey - sy
    raw_length = tl.sqrt(dx * dx + dy * dy)
    length = tl.maximum(raw_length, 1.0e-6)
    cross = dx * (py - sy) - dy * (px - sx)
    correction = tl.where(raw_length >= 1.0e-6, cross / (length * length * length), 0.0)
    alpha = tl.load(OUT + offset, valid, 0.0)
    incoming = tl.load(GRAD + offset, valid, 0.0)
    g = incoming * alpha * (1 - alpha) / SIGMA * sign
    ga_x = g * ((ey - py) / length + correction * dx)
    ga_y = g * ((px - ex) / length + correction * dy)
    gb_x = g * ((py - sy) / length - correction * dx)
    gb_y = g * ((sx - px) / length - correction * dy)
    tl.atomic_add(DUV + (batch * NV + start) * 2, ga_x, valid)
    tl.atomic_add(DUV + (batch * NV + start) * 2 + 1, ga_y, valid)
    tl.atomic_add(DUV + (batch * NV + end) * 2, gb_x, valid)
    tl.atomic_add(DUV + (batch * NV + end) * 2 + 1, gb_y, valid)


class _SoftUnion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uv, faces, valid, size, sigma):
        uv, faces, valid = uv.contiguous(), faces.contiguous(), valid.contiguous()
        if uv.dtype != torch.float32 or not uv.is_cuda:
            raise ValueError("The fused renderer requires float32 CUDA vertices.")
        output = torch.empty((len(uv), size, size), device=uv.device)
        winner = torch.empty_like(output, dtype=torch.int32)
        with torch.cuda.device(uv.device):
            _union_forward[(len(uv), triton.cdiv(size * size, 16))](
                uv,
                faces,
                valid,
                output,
                winner,
                uv.shape[1],
                len(faces),
                size,
                sigma,
                16,
                128,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(uv, faces, output, winner)
        ctx.size, ctx.sigma = size, sigma
        return output

    @staticmethod
    def backward(ctx, grad):
        uv, faces, output, winner = ctx.saved_tensors
        duv = torch.zeros_like(uv)
        with torch.cuda.device(uv.device):
            _union_backward[(len(uv), triton.cdiv(ctx.size * ctx.size, 256))](
                uv,
                faces,
                output,
                winner,
                grad.contiguous(),
                duv,
                uv.shape[1],
                ctx.size,
                ctx.sigma,
                256,
                enable_fp_fusion=False,
            )
        return duv, None, None, None, None


def soft_union(uv, faces, valid, size, sigma):
    return _SoftUnion.apply(uv, faces, valid, size, float(sigma))

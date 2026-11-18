# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Alternating native shape fitting and per-frame articulation optimization."""

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from .geometry import align_points, matrix_rotvec
from .losses import (
    DIMENSION_NAMES,
    constrain_wrist_,
    dimension_targets,
    landmark_loss,
    metric_size_loss,
    native_palm_floor_loss,
    reprojection_loss,
    silhouette_loss,
    smooth_observation_targets,
)
from .preprocess import ObservationStore, select_diverse, surface_training_pool

log = logging.getLogger(__name__)


@dataclass
class FitState:
    beta: np.ndarray
    log_scale: np.ndarray
    bone_logs: np.ndarray
    latent: np.ndarray
    root: np.ndarray
    translation: np.ndarray
    valid: np.ndarray
    failed: np.ndarray


def tensor(a, device):
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def backward_pose_outputs(outputs, leaves):
    """Propagate accumulated image/temporal gradients through native posing once.

    Detached loss leaves let renderer chunks release their graphs immediately.
    Every output contribution is then combined by one exact chain-rule pass;
    parameter priors and translation derivatives accumulate directly as usual.
    """
    active = [
        (output, leaf.grad)
        for output, leaf in zip(outputs, leaves)
        if output.requires_grad and leaf.grad is not None
    ]
    if active:
        torch.autograd.backward([x[0] for x in active], [x[1] for x in active])


def initialize(model, data, side, usable, args):
    device = args.device
    T = data.nframes
    with torch.no_grad():
        _, _, q, _ = model(
            torch.zeros(1, model.nshape, device=device),
            torch.zeros(1, model.npose, device=device),
            torch.zeros(1, 3, device=device),
            torch.zeros(1, 3, device=device),
            torch.zeros(1, device=device),
        )
    template = q[0].cpu().numpy()
    latent = np.zeros((T, model.npose), np.float32)
    root = np.zeros((T, 3), np.float32)
    translation = np.zeros((T, 3), np.float32)
    valid = usable.copy()
    scales = []
    init_weights = np.full(21, 0.05)
    init_weights[[5, 9, 13, 17]] = 1.0
    init_weights[0] = 1.0 if model.wrist_source == "landmark" else 0.0
    if model.wrist_source == "palm":
        # The four MCPs are nearly collinear. The thumb base supplies the
        # off-axis anchor needed to estimate a stable native palm orientation.
        init_weights[1] = 1.0
    for t in np.flatnonzero(valid):
        try:
            R, trans, s = align_points(
                template,
                data.points[t, side],
                init_weights * data.confidence[t, side],
                True,
            )
            root[t], translation[t] = matrix_rotvec(R), trans
            scales.append(s)
        except ValueError:
            valid[t] = False
    scale = (
        float(np.median(scales))
        if scales and args.scale_mode == "metric"
        else args.initial_scale
    )
    for t in np.flatnonzero(valid):
        R, trans, _ = align_points(
            template * scale,
            data.points[t, side],
            init_weights * data.confidence[t, side],
            False,
        )
        root[t], translation[t] = matrix_rotvec(R), trans
    return FitState(
        np.zeros((1, model.nshape), np.float32),
        np.array([np.log(scale)], np.float32),
        np.zeros((1, model.nscale), np.float32),
        latent,
        root,
        translation,
        valid,
        np.zeros(T, bool),
    )


class Fitter:
    def __init__(self, model, renderer, data, side, state, store, args, log_path):
        self.model, self.renderer, self.data, self.side = model, renderer, data, side
        self.state, self.store, self.args = state, store, args
        self.log_path = Path(log_path)
        self.rng = np.random.default_rng(args.seed + side)
        self.scale_anchor = state.log_scale.copy()
        self.frame_quality = np.ones(data.nframes)
        target, weight = dimension_targets(
            data.points[:, side], data.confidence[:, side], state.valid
        )
        self.native_palm_floor_weight = torch.zeros(len(weight), device=args.device)
        self.native_palm_floor_target = tensor(target.copy(), args.device)
        self.use_native_palm_floor = getattr(
            model, "wrist_source", "landmark"
        ) == "palm" and hasattr(model, "native_palm_ratios")
        if self.use_native_palm_floor:
            self.native_palm_floor_target[:4] = model.native_palm_ratios() * float(
                target[4]
            )
            self.native_palm_floor_weight[:4] = float(weight[4])
        if getattr(model, "wrist_source", "landmark") == "palm":
            # An explicitly ignored wrist must not re-enter through metric radii.
            # Palm width and phalanx dimensions still preserve metric hand size.
            weight[:4] = 0
        self.dimension_target = tensor(target, args.device)
        self.dimension_weight = tensor(weight, args.device)
        self.wrist_radius = (
            args.wrist_radius
            if getattr(model, "wrist_source", "landmark") == "landmark"
            else -1
        )
        self._constrain_state_wrist()

    def _constrain_state_wrist(self):
        translation = tensor(self.state.translation.copy(), self.args.device)
        constrain_wrist_(
            translation,
            tensor(self.data.points[:, self.side, 0], self.args.device),
            tensor(
                self.data.confidence[:, self.side, 0] * self.state.valid,
                self.args.device,
            ),
            self.wrist_radius,
        )
        self.state.translation = translation.cpu().numpy()

    def size_loss(self, points):
        loss = metric_size_loss(
            points,
            self.dimension_target,
            self.dimension_weight,
            self.args.metric_size_tolerance,
            self.args.metric_shrink_weight,
        )
        if self.use_native_palm_floor:
            loss = loss + self.args.metric_shrink_weight * native_palm_floor_loss(
                points,
                self.native_palm_floor_target,
                self.native_palm_floor_weight,
                self.args.metric_size_tolerance,
            )
        return loss

    def optimize(self, ids, steps, shape=False, images=True, stage="pose"):
        ids = np.asarray(ids, dtype=int)
        if not len(ids) or steps <= 0:
            return
        a, s, m = self.args, self.state, self.model
        device = a.device
        beta = nn.Parameter(tensor(s.beta.copy(), device), requires_grad=shape)
        scale = nn.Parameter(
            tensor(s.log_scale.copy(), device),
            requires_grad=shape and a.scale_mode == "metric",
        )
        bones = nn.Parameter(
            tensor(s.bone_logs.copy(), device), requires_grad=shape and a.optimize_bones
        )
        latent = nn.Parameter(
            tensor(s.latent[ids].copy(), device), requires_grad=not shape
        )
        root = nn.Parameter(tensor(s.root[ids].copy(), device), requires_grad=not shape)
        trans = nn.Parameter(
            tensor(s.translation[ids].copy(), device), requires_grad=not shape
        )
        groups = (
            [
                {
                    "params": [x for x in (beta, scale, bones) if x.requires_grad],
                    "lr": a.shape_lr,
                }
            ]
            if shape
            else [
                {"params": [latent, root], "lr": a.pose_lr},
                {"params": [trans], "lr": a.translation_lr},
            ]
        )
        optimizer = torch.optim.Adam(groups)
        params = [beta, scale, bones, latent, root, trans]
        observed = tensor(self.data.points[ids, self.side], device)
        wrist_confidence = tensor(self.data.confidence[ids, self.side, 0], device)
        confidence = (
            tensor(self.data.confidence[ids, self.side], device) * m.observation_weights
        )
        cameras = (
            tensor(self.data.K[ids], device)
            if images and a.reprojection_weight > 0
            else None
        )
        scale_anchor = tensor(self.scale_anchor, device)
        obs = self.store.load(ids) if images else None
        best, best_value = None, float("inf")
        records = []
        last_finite = [p.detach().clone() for p in params]
        if not shape:
            with torch.no_grad():
                m.prepare_identity(
                    beta,
                    scale,
                    bones if a.optimize_bones else None,
                )
        # Keep rendering sharpness fixed so saved iterates have comparable losses.
        # Shape uses stochastic batches; pose retains the best fixed-objective iterate.
        for iteration in range(steps + 1):
            sampling = np.maximum(self.frame_quality[ids], 1e-4) ** 0.5
            sampling /= sampling.sum()
            local = (
                self.rng.choice(
                    len(ids), min(a.batch_size, len(ids)), replace=False, p=sampling
                )
                if shape
                else np.arange(len(ids))
            )
            ix = torch.as_tensor(local, device=device)
            optimizer.zero_grad(set_to_none=True)
            if shape:
                vertices, _joints, q, R = m(
                    beta,
                    latent[ix],
                    root[ix],
                    trans[ix],
                    scale,
                    bones if a.optimize_bones else None,
                )
            else:
                vertices, _joints, q, R = m.pose_prepared(
                    latent[ix], root[ix], trans[ix]
                )
            key = landmark_loss(
                q, observed[ix], confidence[ix], a.landmark_tolerance, a.landmark_scale
            ).mean()
            reprojection = vertices.sum() * 0
            if images and a.reprojection_weight > 0:
                reprojection = reprojection_loss(
                    q,
                    observed[ix],
                    confidence[ix],
                    cameras[ix],
                    a.reprojection_tolerance,
                    a.reprojection_scale,
                ).mean()
            size_loss = self.size_loss(q).mean()
            image = vertices.sum() * 0
            if images:
                near = vertices[..., 2].amin()
                if near.detach().item() > 0.001:
                    pred = self.renderer(
                        vertices,
                        m.faces,
                        obs["K"][ix],
                        sigma=0.3,
                    )
                    per_image = silhouette_loss(
                        pred,
                        obs["mask"][ix],
                        obs["weight"][ix],
                        obs["distance"][ix],
                        obs["balance"][ix],
                        getattr(a, "surface_coverage_weight", 2.0),
                        obs["wrist_region"][ix],
                        getattr(a, "wrist_surface_weight", 2.0),
                    )
                    has_mask = (obs["weight"][ix].sum((-1, -2)) > 0).to(pred.dtype)
                    image = (per_image * has_mask).sum() / has_mask.sum().clamp_min(1)
            prior = a.shape_prior * beta.square().mean() + a.pose_prior * latent[
                ix
            ].square().sum(-1).mean() / min(latent.shape[-1], 32)
            prior = (
                prior + a.scale_prior * ((scale - scale_anchor) / 0.25).square().mean()
            )
            if a.optimize_bones:
                prior = prior + a.bone_prior * (bones / 0.15).square().mean()
            depth = torch.relu(0.02 - vertices[..., 2]).square().mean() * 1000
            temporal = vertices.sum() * 0
            if not shape and a.temporal_weight > 0:
                dt = np.diff(self.data.timestamps[ids])
                adjacent = (np.diff(ids) == 1) & (dt <= a.temporal_max_gap)
                if adjacent.any():
                    pair = torch.as_tensor(np.flatnonzero(adjacent), device=device)
                    factor = tensor((1 / 30) / dt[adjacent], device)
                    temporal = (
                        ((R[pair + 1] - R[pair]) * factor[:, None, None, None])
                        .square()
                        .mean()
                    )
                    temporal += (
                        (((trans[pair + 1] - trans[pair]) / 0.05) * factor[:, None])
                        .square()
                        .mean()
                    )
                first = int(ids[0])
                if (
                    first > 0
                    and s.valid[first - 1]
                    and self.data.timestamps[first] - self.data.timestamps[first - 1]
                    <= a.temporal_max_gap
                ):
                    prev_R = m.decode(
                        tensor(s.latent[first - 1 : first], device),
                        tensor(s.root[first - 1 : first], device),
                    ).detach()
                    factor = (1 / 30) / (
                        self.data.timestamps[first] - self.data.timestamps[first - 1]
                    )
                    temporal += ((R[:1] - prev_R) * factor).square().mean()
                    temporal += (
                        (
                            (
                                trans[:1]
                                - tensor(s.translation[first - 1 : first], device)
                            )
                            / 0.05
                            * factor
                        )
                        .square()
                        .mean()
                    )
            loss = (
                a.keypoint_weight * key
                + a.image_weight * image
                + a.reprojection_weight * reprojection
                + a.metric_size_weight * size_loss
                + prior
                + depth
                + a.temporal_weight * temporal
            )
            value = float(loss.detach())
            if not np.isfinite(value):
                s.failed[ids] = True
                log.warning(
                    "Nonfinite objective during %s on %s; retaining previous/best values",
                    stage,
                    ids.tolist(),
                )
                break
            if not shape and value < best_value:
                best_value = value
                best = [p.detach().clone() for p in params]
            last_finite = [p.detach().clone() for p in params]
            if iteration in (0, steps):
                records.append(
                    {
                        "stage": stage,
                        "iteration": iteration,
                        "frames": ids.tolist(),
                        "loss": value,
                        "keypoint": float(key.detach()),
                        "image": float(image.detach()),
                        "metric_size": float(size_loss.detach()),
                    }
                )
            # Evaluate the final Adam update as well; steps=1 must perform one update.
            if iteration == steps:
                break
            loss.backward()
            trainable = [p for p in params if p.requires_grad]
            if any(
                p.grad is None or not torch.isfinite(p.grad).all() for p in trainable
            ):
                s.failed[ids] = True
                log.warning(
                    "Invalid gradient during %s; retaining previous/best values", stage
                )
                break
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            with torch.no_grad():
                beta.clamp_(-3, 3)
                scale.clamp_(np.log(0.3), np.log(3))
                if m.identity_backend == "mhr":
                    bones.clamp_(-a.mhr_scale_limit, a.mhr_scale_limit)
                else:
                    bones.clamp_(np.log(0.7), np.log(1.3))
                latent.clamp_(-4, 4)
                trans[:, 2].clamp_(min=0.03)
                constrain_wrist_(
                    trans, observed[:, 0], wrist_confidence, self.wrist_radius
                )
        if shape:
            with torch.no_grad():
                for p, b in zip(params, last_finite):
                    p.copy_(b)
        if not shape and best is not None:
            with torch.no_grad():
                for p, b in zip(params, best):
                    p.copy_(b)
        s.beta = beta.detach().cpu().numpy()
        s.log_scale = scale.detach().cpu().numpy()
        s.bone_logs = bones.detach().cpu().numpy()
        s.latent[ids] = latent.detach().cpu().numpy()
        s.root[ids] = root.detach().cpu().numpy()
        s.translation[ids] = trans.detach().cpu().numpy()
        with self.log_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def run_pose_batches(self, ids, steps, images, stage):
        ids = np.sort(ids)
        # Landmark-only fitting does not allocate rasterizer intermediates.
        # Use larger batches and retain temporal links inside each batch.
        batch_size = (
            self.args.batch_size
            if images
            else max(self.args.batch_size, self.args.keypoint_batch_size)
        )
        for start in tqdm(range(0, len(ids), batch_size), desc=stage):
            self.optimize(ids[start : start + batch_size], steps, False, images, stage)

    def _temporal_context(self, ids, device):
        dt = np.diff(self.data.timestamps[ids])
        adjacent = (np.diff(ids) == 1) & (dt > 0) & (dt <= self.args.temporal_max_gap)
        context = {
            "pairs": torch.as_tensor(np.flatnonzero(adjacent), device=device),
            "triples": torch.as_tensor(
                np.flatnonzero(adjacent[:-1] & adjacent[1:]), device=device
            ),
            "intervals": tensor(dt, device),
            "quadruples": torch.as_tensor(
                np.flatnonzero(adjacent[:-2] & adjacent[1:-1] & adjacent[2:]),
                device=device,
            ),
        }
        mode = getattr(self.args, "temporal_target", "stationary")
        if mode in ("observations", "smooth_observations"):
            target = self.data.points[ids, self.side].copy()
            confidence = self.data.confidence[ids, self.side].copy()
            good = (
                np.isfinite(target).all(-1)
                & (target[..., 2] > 0)
                & np.isfinite(confidence)
            )
            confidence = np.where(good, confidence, 0)
            if (
                getattr(getattr(self, "model", None), "wrist_source", "landmark")
                == "palm"
            ):
                confidence[:, 0] = 0
            if mode == "smooth_observations":
                target = smooth_observation_targets(
                    target,
                    confidence,
                    self.data.timestamps[ids],
                    ids,
                    self.args.temporal_max_gap,
                    getattr(self.args, "temporal_smoothing_window", 0.067),
                    getattr(self.args, "temporal_max_correction", 0.002),
                )
            context["target"] = tensor(np.nan_to_num(target), device)
            confidence = tensor(confidence, device)
            t = context["triples"]
            context["weight"] = torch.minimum(
                torch.minimum(confidence[t], confidence[t + 1]), confidence[t + 2]
            )
        return context

    def _sequence_temporal_loss(
        self,
        rotations,
        translations,
        ids,
        joints=None,
        observations=None,
        context=None,
        vertices=None,
    ):
        """Time-aware derivatives, normalized to the original 30 Hz penalty units.

        Only consecutive valid frames within temporal_max_gap are coupled.
        Unequal intervals use velocity differences over the midpoint interval.
        """
        zero = rotations.sum() * 0
        if len(ids) < 2:
            return zero
        context = (
            self._temporal_context(ids, rotations.device)
            if context is None
            else context
        )
        pair = context["pairs"]
        if not len(pair):
            return zero
        intervals = context["intervals"].to(dtype=rotations.dtype)

        def velocity(values, indices):
            factor = (1 / 30) / intervals[indices]
            return (values[indices + 1] - values[indices]) * factor.reshape(
                -1, *([1] * (values.ndim - 1))
            )

        result = velocity(rotations, pair).square().mean()
        result = result + (velocity(translations, pair) / 0.05).square().mean()
        triples = context["triples"]
        if len(triples):

            def acceleration(values):
                factor = (2 / 30) / (intervals[triples] + intervals[triples + 1])
                return (
                    velocity(values, triples + 1) - velocity(values, triples)
                ) * factor.reshape(-1, *([1] * (values.ndim - 1)))

            result = result + 4 * acceleration(rotations).square().mean()
            result = result + (acceleration(translations) / 0.005).square().mean()
            if observations is not None and "target" in context:
                residual = observations - context["target"]
                triple_weight = context["weight"]
                error = (acceleration(residual) / 0.002).square().mean(-1)
                result = result + (
                    error * triple_weight
                ).sum() / triple_weight.sum().clamp_min(1)
            elif joints is not None:
                result = result + (acceleration(joints) / 0.002).square().mean()
        mesh_weight = getattr(self.args, "temporal_mesh_jerk_weight", 0.0)
        root_weight = getattr(self.args, "temporal_root_jerk_weight", 0.0)
        quadruples = context["quadruples"]
        if len(quadruples) and (mesh_weight > 0 or root_weight > 0):
            # Surface motion constrains twists invisible to sparse joints. The
            # root term prevents absorbing surface jitter into wrist translation.
            # Third derivatives preserve constant-acceleration motion.
            i = quadruples
            left_dt = (intervals[i] + intervals[i + 1]) / 2
            right_dt = (intervals[i + 1] + intervals[i + 2]) / 2
            jerk_dt = (intervals[i] + 2 * intervals[i + 1] + intervals[i + 2]) / 4

            def jerk(values):
                shape = (-1,) + (1,) * (values.ndim - 1)
                left_acc = (velocity(values, i + 1) - velocity(values, i)) / (
                    30 * left_dt.reshape(shape)
                )
                right_acc = (velocity(values, i + 2) - velocity(values, i + 1)) / (
                    30 * right_dt.reshape(shape)
                )
                return (right_acc - left_acc) / (30 * jerk_dt.reshape(shape))

            if vertices is not None and mesh_weight > 0:
                result = result + mesh_weight * (jerk(vertices) / 0.002).square().mean()
            if root_weight > 0:
                result = (
                    result + root_weight * (jerk(translations) / 0.002).square().mean()
                )
        return result

    def optimize_sequence(
        self, ids, steps, images=True, stage="sequence_pose", active_ids=None
    ):
        """Optimize one pose trajectory while chunking only renderer evaluation.

        Every valid frame belongs to one optimizer and temporal terms span chunk
        boundaries. Chunking controls memory only.
        """
        ids = np.sort(np.asarray(ids, dtype=int))
        if not len(ids) or steps <= 0:
            return
        a, s, m = self.args, self.state, self.model
        device = a.device
        active = None
        if active_ids is not None:
            active = torch.as_tensor(np.isin(ids, active_ids), device=device)
            if not active.any():
                return
        beta = tensor(s.beta, device)
        scale = tensor(s.log_scale, device)
        bones = tensor(s.bone_logs, device)
        # Start from the actual fitted trajectory. Averaging axis-angle vectors
        # can turn equivalent +pi/-pi rotations into an unrelated identity pose,
        # and pre-smoothing would discard our rollback baseline.
        latent = nn.Parameter(tensor(s.latent[ids].copy(), device))
        root = nn.Parameter(tensor(s.root[ids].copy(), device))
        trans = nn.Parameter(tensor(s.translation[ids].copy(), device))
        optimizer = torch.optim.Adam(
            [
                {"params": [latent, root], "lr": a.pose_lr},
                {"params": [trans], "lr": a.translation_lr},
            ]
        )
        observed = tensor(self.data.points[ids, self.side], device)
        wrist_confidence = tensor(self.data.confidence[ids, self.side, 0], device)
        confidence = (
            tensor(self.data.confidence[ids, self.side], device) * m.observation_weights
        )
        cameras = (
            tensor(self.data.K[ids], device)
            if images and a.reprojection_weight > 0
            else None
        )
        sequence_batch_size = getattr(a, "sequence_batch_size", a.batch_size)
        chunks = [
            np.arange(start, min(start + sequence_batch_size, len(ids)))
            for start in range(0, len(ids), sequence_batch_size)
        ]
        mask_count = 0
        if images:
            for local in chunks:
                obs = self.store.load(ids[local])
                mask_count += int((obs["weight"].sum((-1, -2)) > 0).sum().item())
        mask_count = max(mask_count, 1)
        params = [latent, root, trans]
        frozen = [p.detach().clone() for p in params] if active is not None else None
        best = [p.detach().clone() for p in params]
        best_value = float("inf")
        records = []
        with torch.no_grad():
            m.prepare_identity(
                beta,
                scale,
                bones if a.optimize_bones else None,
            )
        temporal_context = self._temporal_context(ids, device)
        for iteration in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(3, device=device)
            # Pose every frame once. Accumulate spatial and temporal gradients
            # at its outputs without retaining any renderer chunk's graph.
            pose_outputs = m.pose_prepared(latent, root, trans)
            pose_leaves = tuple(
                value.detach().requires_grad_(True) for value in pose_outputs
            )
            sequence_vertices, sequence_joints, sequence_observations, rotations = (
                pose_leaves
            )
            for local in chunks:
                ix = slice(int(local[0]), int(local[-1]) + 1)
                vertices, q = sequence_vertices[ix], sequence_observations[ix]
                key = landmark_loss(
                    q,
                    observed[ix],
                    confidence[ix],
                    a.landmark_tolerance,
                    a.landmark_scale,
                ).sum() / len(ids)
                reprojection = vertices.sum() * 0
                if images and a.reprojection_weight > 0:
                    reprojection = reprojection_loss(
                        q,
                        observed[ix],
                        confidence[ix],
                        cameras[ix],
                        a.reprojection_tolerance,
                        a.reprojection_scale,
                    ).sum() / len(ids)
                image = vertices.sum() * 0
                if images:
                    obs = self.store.load(ids[local])
                    pred = self.renderer(
                        vertices,
                        m.faces,
                        obs["K"],
                        sigma=0.3,
                    )
                    per_image = silhouette_loss(
                        pred,
                        obs["mask"],
                        obs["weight"],
                        obs["distance"],
                        obs["balance"],
                        getattr(a, "surface_coverage_weight", 2.0),
                        obs["wrist_region"],
                        getattr(a, "wrist_surface_weight", 2.0),
                    )
                    has_mask = (obs["weight"].sum((-1, -2)) > 0).to(pred.dtype)
                    # Reject only an invalid frame, independently of batch size.
                    in_front = (vertices[..., 2].amin(-1) > 0.001).to(pred.dtype)
                    image = (per_image * has_mask * in_front).sum() / mask_count
                prior = (
                    a.pose_prior
                    * latent[ix].square().sum()
                    / (len(ids) * min(latent.shape[-1], 32))
                )
                depth = (
                    torch.relu(0.02 - vertices[..., 2]).square().mean()
                    * 1000
                    * len(local)
                    / len(ids)
                )
                chunk_loss = (
                    a.keypoint_weight * key
                    + a.image_weight * image
                    + a.reprojection_weight * reprojection
                    + prior
                    + depth
                    + a.metric_size_weight * self.size_loss(q).sum() / len(ids)
                )
                if iteration < steps:
                    chunk_loss.backward()
                totals += torch.stack(
                    (chunk_loss.detach(), key.detach(), image.detach())
                )
            temporal = self._sequence_temporal_loss(
                rotations,
                trans,
                ids,
                sequence_joints,
                sequence_observations,
                temporal_context,
                sequence_vertices,
            )
            totals[0] += a.temporal_weight * temporal.detach()
            # One synchronization per iteration instead of several per chunk.
            total_value, key_value, image_value, temporal_value = (
                torch.cat((totals, temporal.detach().reshape(1))).cpu().tolist()
            )
            if not np.isfinite(total_value):
                s.failed[ids] = True
                log.warning(
                    "Nonfinite objective during %s; retaining best values", stage
                )
                break
            if total_value < best_value:
                best_value = total_value
                best = [p.detach().clone() for p in params]
            if iteration in (0, steps):
                records.append(
                    {
                        "stage": stage,
                        "iteration": iteration,
                        "frames": ids.tolist(),
                        "loss": total_value,
                        "keypoint": key_value,
                        "image": image_value,
                        "temporal": temporal_value,
                    }
                )
            if iteration == steps:
                break
            (a.temporal_weight * temporal).backward()
            backward_pose_outputs(pose_outputs, pose_leaves)
            if (
                any(p.grad is None for p in params)
                or not torch.stack([torch.isfinite(p.grad).all() for p in params]).all()
            ):
                s.failed[ids] = True
                log.warning("Invalid gradient during %s; retaining best values", stage)
                break
            if active is not None:
                for parameter in params:
                    parameter.grad[~active] = 0
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            multiplier = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (iteration + 1) / steps))
            for group, initial_lr in zip(
                optimizer.param_groups, (a.pose_lr, a.translation_lr)
            ):
                group["lr"] = initial_lr * multiplier
            with torch.no_grad():
                latent.clamp_(-4, 4)
                trans[:, 2].clamp_(min=0.03)
                constrain_wrist_(
                    trans, observed[:, 0], wrist_confidence, self.wrist_radius
                )
                if active is not None:
                    for parameter, original in zip(params, frozen):
                        parameter[~active] = original[~active]
        with torch.no_grad():
            for parameter, value in zip(params, best):
                parameter.copy_(value)
        s.latent[ids] = latent.detach().cpu().numpy()
        s.root[ids] = root.detach().cpu().numpy()
        s.translation[ids] = trans.detach().cpu().numpy()
        with self.log_path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def polish_sequence(self, ids, steps):
        """Spend a small part of the saved runtime on a finer pose-only finish."""
        if steps <= 0:
            return
        pose_lr, translation_lr = self.args.pose_lr, self.args.translation_lr
        try:
            self.args.pose_lr = pose_lr * 0.1
            self.args.translation_lr = translation_lr * 0.1
            self.optimize_sequence(ids, steps, images=True, stage="fine_sequence_pose")
        finally:
            self.args.pose_lr, self.args.translation_lr = pose_lr, translation_lr

    @torch.no_grad()
    def evaluate(self, ids):
        result = {}
        s, m, a = self.state, self.model, self.args
        m.prepare_identity(
            tensor(s.beta, a.device),
            tensor(s.log_scale, a.device),
            tensor(s.bone_logs, a.device) if a.optimize_bones else None,
        )
        for t in ids:
            obs = self.store.load([t])
            v, _j, _q, _R = m.pose_prepared(
                tensor(s.latent[t : t + 1], a.device),
                tensor(s.root[t : t + 1], a.device),
                tensor(s.translation[t : t + 1], a.device),
            )
            if v[..., 2].amin() <= 0.001:
                result[int(t)] = {"iou": 0.0, "surface_loss": 10.0}
                continue
            pred = self.renderer(v, m.faces, obs["K"])
            p = pred > 0.5
            target = obs["mask"] > 0.5
            w = obs["weight"] > 0.5
            union = ((p | target) & w).sum().item()
            iou = float(((p & target) & w).sum().item() / union) if union else None
            loss = (
                float(
                    silhouette_loss(
                        pred,
                        obs["mask"],
                        obs["weight"],
                        obs["distance"],
                        obs["balance"],
                        getattr(a, "surface_coverage_weight", 2.0),
                        obs["wrist_region"],
                        getattr(a, "wrist_surface_weight", 2.0),
                    )[0]
                )
                if w.any()
                else None
            )
            result[int(t)] = {"iou": iou, "surface_loss": loss}
        return result

    def save_checkpoint(self, path):
        s = self.state
        np.savez_compressed(
            path,
            beta=s.beta,
            log_scale=s.log_scale,
            bone_logs=s.bone_logs,
            latent=s.latent,
            root=s.root,
            translation=s.translation,
            valid=s.valid,
            failed=s.failed,
        )


def fit_side(
    model, renderer, data, side, quality, descriptors, usable, cache, args, outdir
):
    if (
        args.wrist_source == "landmark"
        and args.wrist_radius >= 0
        and int(model.joint_map[0]) != 0
    ):
        raise ValueError(
            "Bounded wrist fitting requires input point 0 to map to native root 0."
        )
    state = initialize(model, data, side, usable, args)
    ids = np.flatnonzero(state.valid)
    if not len(ids):
        return state, {
            "status": "no_valid_observations",
            "selected_frames": [],
            "holdout_frames": [],
        }
    store = ObservationStore(
        cache,
        ("left", "right")[side],
        args.device,
        host_cache_mib=getattr(args, "observation_host_cache_mib", 256),
        device_cache_mib=getattr(args, "observation_device_cache_mib", 128),
    )
    fitter = Fitter(
        model,
        renderer,
        data,
        side,
        state,
        store,
        args,
        Path(outdir) / f"{model.side}_optimization.jsonl",
    )
    fitter.frame_quality = quality.copy()
    fitter.run_pose_batches(
        ids, args.init_steps, False, f"{model.side}: 3D initialization"
    )
    if args.metric_size_weight > 0 and args.metric_shape_steps > 0:
        # Shared dimensions come from every usable input frame, independent of masks.
        metric_ids = ids[np.linspace(0, len(ids) - 1, min(32, len(ids)), dtype=int)]
        fitter.optimize(
            metric_ids, args.metric_shape_steps, True, False, "metric_shape"
        )
        fitter.run_pose_batches(
            ids, args.pose_steps, False, f"{model.side}: metric pose refinement"
        )
    eligible = state.valid & (quality > 0) & np.isfinite(descriptors).all(1)
    candidates = np.flatnonzero(eligible)
    holdout = candidates[::5] if len(candidates) >= 10 else np.array([], int)
    if len(holdout) > args.holdout_limit:
        holdout = holdout[
            np.linspace(0, len(holdout) - 1, args.holdout_limit, dtype=int)
        ]
    training = eligible.copy()
    training[holdout] = False
    with np.load(Path(cache) / "index.npz") as index:
        visibility = (
            np.array(index["surface_visibility"][:, side])
            if "surface_visibility" in index
            else np.ones(data.nframes)
        )
    training, shape_quality = surface_training_pool(
        training, quality, visibility, getattr(args, "shape_visibility_fraction", 0)
    )
    fitter.frame_quality = shape_quality
    selected = select_diverse(descriptors, shape_quality, args.keyframes, training)
    if not len(selected):
        log.warning(
            "No trustworthy segmented frames for %s: output is keypoint-only.",
            model.side,
        )
    fitter.run_pose_batches(
        holdout, args.final_steps, True, f"{model.side}: baseline holdout poses"
    )
    before = fitter.evaluate(holdout)
    metric_targets = {
        name: {"meters": float(value), "weight": float(weight)}
        for name, value, weight in zip(
            DIMENSION_NAMES,
            fitter.dimension_target.cpu(),
            fitter.dimension_weight.cpu(),
        )
    }
    if len(selected):
        fitter.run_pose_batches(
            selected, args.pose_steps, True, f"{model.side}: align selected frames"
        )
        for round_index in range(args.rounds):
            log.info(
                "%s: shared shape round %d/%d (%d keyframes)",
                model.side,
                round_index + 1,
                args.rounds,
                len(selected),
            )
            fitter.optimize(selected, args.shape_steps, True, True, "shared_shape")
            fitter.run_pose_batches(
                selected, args.pose_steps, True, f"{model.side}: refine selected poses"
            )
            fitter.save_checkpoint(Path(outdir) / f"{model.side}_checkpoint.npz")
        # Evaluate a bounded diverse pool after pose alignment, then add credible high residual views.
        pool = select_diverse(
            descriptors, shape_quality, args.residual_candidates, training
        )
        pool = np.setdiff1d(pool, selected)
        added = np.array([], int)
        if len(pool) and args.add_keyframes > 0:
            fitter.run_pose_batches(
                pool, args.pose_steps, True, f"{model.side}: residual candidates"
            )
            scores = fitter.evaluate(pool)
            residual_quality = np.zeros(data.nframes)
            for t in pool:
                metric = scores[int(t)]
                if (
                    metric["iou"] is not None
                    and metric["iou"] >= 0.25
                    and metric["surface_loss"] is not None
                ):
                    residual_quality[t] = quality[t] * min(metric["surface_loss"], 2)
            added = select_diverse(
                descriptors, residual_quality, args.add_keyframes, training
            )
            if len(added):
                selected = np.union1d(selected, added)
                fitter.optimize(
                    selected, args.shape_steps, True, True, "shared_shape_extra_views"
                )
    else:
        added = np.array([], int)
    refine_steps = getattr(args, "surface_refine_steps", 0) if len(selected) else 0
    final_pose_steps = min(args.final_steps // 2, 40) if refine_steps else 0
    fitter.optimize_sequence(
        ids,
        args.final_steps - final_pose_steps,
        True,
        f"{model.side}: whole-sequence fit",
    )
    if refine_steps:
        # Shape must see the final aligned poses, including visible wrist sides.
        # Reuse selected training views; holdout images never fit shared shape.
        fitter.optimize(selected, refine_steps, True, True, "late_surface_shape")
        fitter.optimize_sequence(
            ids, final_pose_steps, True, f"{model.side}: final surface pose fit"
        )
    fitter.polish_sequence(ids, getattr(args, "sequence_polish_steps", 0))
    after = fitter.evaluate(holdout)
    fitter.save_checkpoint(Path(outdir) / f"{model.side}_checkpoint.npz")
    return state, {
        "status": "surface_and_keypoints" if len(selected) else "keypoint_only",
        "selected_frames": selected.tolist(),
        "surface_training_frames": np.flatnonzero(training).tolist(),
        "selected_surface_visibility": visibility[selected].tolist(),
        "native_palm_floor": {
            "enabled": fitter.use_native_palm_floor,
            "reference": "native neutral palm radii scaled by observed MCP width; source wrist excluded",
            "radii_m": fitter.native_palm_floor_target[:4].cpu().tolist()
            if fitter.use_native_palm_floor
            else None,
        },
        "metric_dimension_targets": metric_targets,
        "added_frames": added.tolist(),
        "holdout_frames": holdout.tolist(),
        "holdout_before": before,
        "holdout_after": after,
        "numerical_warning_frames": np.flatnonzero(state.failed).tolist(),
        "holdout_note": "Shape never sees holdout images; holdout poses are fitted to their own observations. No 3D accuracy claim.",
    }

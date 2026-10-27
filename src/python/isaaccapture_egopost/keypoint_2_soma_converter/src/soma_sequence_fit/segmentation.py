# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Image-derived masks with automatic left/right association and explicit fallbacks."""

from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import FINGERS, project_np


def mask_score(mask, uv, confidence):
    h, w = mask.shape
    good = np.isfinite(uv).all(-1) & (confidence > 0.2)
    if good.sum() < 3 or mask.sum() < 10:
        return 0.0
    xy = np.rint(uv[good]).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    hit = np.zeros(len(xy), float)
    hit[inside] = mask[xy[inside, 1], xy[inside, 0]]
    low, high = uv[good].min(0), uv[good].max(0)
    ys, xs = np.where(mask)
    lo, hi = np.array([xs.min(), ys.min()]), np.array([xs.max(), ys.max()])
    inter = np.maximum(0, np.minimum(high, hi) - np.maximum(low, lo)).prod()
    union = np.maximum(high - low, 1).prod() + np.maximum(hi - lo, 1).prod() - inter
    return float(0.7 * hit.mean() + 0.3 * inter / max(union, 1))


class MaskProvider:
    def __init__(self, backend, mask_dir=None, device="cuda", checkpoint=None):
        self.backend = backend
        self.mask_dir = Path(mask_dir) if mask_dir else None
        if backend == "files" and not self.mask_dir:
            raise ValueError("--mask-dir is required for the files backend.")
        if backend == "sam3":
            if not str(device).startswith("cuda"):
                raise ValueError("SAM3 requires CUDA. Use files or grabcut on CPU.")
            checkpoint = checkpoint or "facebook/sam3"
            if Path(checkpoint).is_file():
                raise ValueError(
                    "--sam3-checkpoint must be a Hugging Face model ID or a local "
                    "save_pretrained directory, not a standalone checkpoint file."
                )
            try:
                from transformers import Sam3Model, Sam3Processor
            except ImportError as e:
                raise ImportError(
                    "Install the project dependencies from the project directory: "
                    "python -m pip install -r requirements.txt."
                ) from e
            import torch

            self.device = torch.device(device)
            self.model = (
                Sam3Model.from_pretrained(checkpoint, attn_implementation="sdpa")
                .to(self.device)
                .eval()
            )
            self.processor = Sam3Processor.from_pretrained(checkpoint)
            text = self.processor(text="hand", return_tensors="pt").to(self.device)
            self.attention_mask = text.attention_mask
            # The prompt is fixed across the clip; encode it only once.
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.text_embeds = self.model.get_text_features(**text)

    def __call__(self, frame, index, points, confidence, K):
        h, w = frame.shape[:2]
        uv = [project_np(points[s], K) for s in range(2)]
        result = [np.zeros((h, w), np.uint8) for _ in range(2)]
        scores = [0.0, 0.0]
        if self.backend == "files":
            for s, side in enumerate(("left", "right")):
                path = self.mask_dir / side / f"{index:06d}.png"
                if not path.exists():
                    continue
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (h, w):
                    raise ValueError(f"Mask must match full video resolution: {path}")
                result[s] = (mask > 127).astype(np.uint8)
                scores[s] = mask_score(result[s], uv[s], confidence[s])
            return result, scores
        if self.backend == "grabcut":
            for s in range(2):
                good = np.isfinite(uv[s]).all(1) & (confidence[s] > 0.2)
                if good.sum() < 6:
                    continue
                pts = uv[s][good]
                lo, hi = pts.min(0), pts.max(0)
                margin = max(float((hi - lo).max()) * 0.25, 8)
                x0, y0 = np.maximum(np.floor(lo - margin), 0).astype(int)
                x1, y1 = np.minimum(np.ceil(hi + margin), [w - 1, h - 1]).astype(int)
                if x1 - x0 < 5 or y1 - y0 < 5:
                    continue
                roi = np.s_[y0 : y1 + 1, x0 : x1 + 1]
                patch = frame[roi]
                shape = patch.shape[:2]
                origin = np.array([x0, y0])
                labels = np.full(shape, cv2.GC_PR_BGD, np.uint8)
                seeds = np.zeros(shape, np.uint8)
                thickness = max(2, round(max(x1 - x0, y1 - y0) * 0.025))
                palm_ids = [0, 5, 9, 13, 17]
                if good[palm_ids].all():
                    palm = np.rint(uv[s][palm_ids]).astype(np.int32) - origin.astype(
                        np.int32
                    )
                    cv2.fillConvexPoly(seeds, cv2.convexHull(palm), 255)
                for finger in FINGERS:
                    chain = [0] + finger
                    for a, b in zip(chain[:-1], chain[1:]):
                        if good[a] and good[b]:
                            cv2.line(
                                seeds,
                                tuple(np.rint(uv[s][a]).astype(int) - origin),
                                tuple(np.rint(uv[s][b]).astype(int) - origin),
                                255,
                                thickness,
                                cv2.LINE_AA,
                            )
                for p in pts:
                    if 0 <= p[0] < w and 0 <= p[1] < h:
                        cv2.circle(
                            seeds,
                            tuple(np.rint(p).astype(int) - origin),
                            thickness,
                            255,
                            -1,
                        )

                # Learn a broad color envelope from the palm itself. This only
                # removes implausible seeds and definite background; GrabCut
                # still decides the uncertain boundary and can retain shadows.
                palm_region = np.zeros(shape, np.uint8)
                if good[palm_ids].all():
                    cv2.fillConvexPoly(palm_region, cv2.convexHull(palm), 255)
                palm_core = cv2.erode(palm_region, np.ones((5, 5), np.uint8))
                color_distance = None
                if palm_core.sum() >= 20 * 255:
                    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
                    center = np.median(lab[palm_core > 0], axis=0)
                    color_distance = np.sqrt(
                        (((lab - center) / [48, 20, 20]) ** 2).sum(2)
                    )
                    seeds[color_distance > 2.2] = 0

                # Extend the search envelope across the proximal palm. Source
                # point 0 is a joint center, not an observed skin cut. This is
                # probable foreground only; image color still decides the edge.
                if good[palm_ids].all():
                    wrist = uv[s][0] - origin
                    forward = uv[s][palm_ids[1:]].mean(0) - uv[s][0]
                    length = np.linalg.norm(forward)
                    if length > 2:
                        direction = forward / length
                        lateral = np.array([-direction[1], direction[0]])
                        width = np.linalg.norm(uv[s][5] - uv[s][17])
                        proximal = wrist - 0.4 * forward
                        band = np.array(
                            [
                                wrist - 0.45 * width * lateral,
                                wrist + 0.45 * width * lateral,
                                proximal + 0.45 * width * lateral,
                                proximal - 0.45 * width * lateral,
                            ]
                        )
                        cv2.fillConvexPoly(seeds, np.rint(band).astype(np.int32), 255)

                halo_size = max(5, 2 * round(max(x1 - x0, y1 - y0) * 0.06) + 1)
                halo = cv2.dilate(seeds, np.ones((halo_size, halo_size), np.uint8))
                # Geometry supplies a generous search envelope, not a hard hand
                # mask. Background outside it must remain background; otherwise
                # a similarly colored table can become one connected hand region.
                labels[halo == 0] = cv2.GC_BGD
                labels[halo > 0] = cv2.GC_PR_FGD
                # Projected finger points may miss the visible hand by pixels.
                # Forcing their circles to foreground contaminates the learned
                # colors and prevents GrabCut from correcting those boundaries.
                labels[palm_core > 0] = cv2.GC_FGD
                if not np.any(labels == cv2.GC_FGD):
                    continue
                # One-pixel interior shafts retain illumination changes along
                # the fingers without forcing broad disks around uncertain tips.
                if color_distance is not None:
                    for finger in FINGERS:
                        for index, (a, b) in enumerate(zip(finger[:-1], finger[1:])):
                            if not (good[a] and good[b]):
                                continue
                            pa = uv[s][a] - origin
                            pb = (
                                uv[s][b]
                                if index < 2
                                else 0.1 * uv[s][a] + 0.9 * uv[s][b]
                            ) - origin
                            cv2.line(
                                labels,
                                tuple(np.rint(pa).astype(int)),
                                tuple(np.rint(pb).astype(int)),
                                cv2.GC_FGD,
                                1,
                            )
                    # Negative seeds in genuinely open interdigital gaps. These
                    # are kept away from every observed shaft and the palm core.
                    shafts = np.zeros(shape, np.uint8)
                    for finger in FINGERS:
                        for a, b in zip(finger[:-1], finger[1:]):
                            if good[a] and good[b]:
                                cv2.line(
                                    shafts,
                                    tuple(np.rint(uv[s][a] - origin).astype(int)),
                                    tuple(np.rint(uv[s][b] - origin).astype(int)),
                                    1,
                                    1,
                                )
                    away = cv2.distanceTransform(
                        1 - shafts, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
                    )
                    safe_gap = max(3.0, max(x1 - x0, y1 - y0) * 0.025)
                    for first, second in zip(FINGERS[1:-1], FINGERS[2:]):
                        if not good[first + second].all():
                            continue
                        base = (uv[s][first[1]] + uv[s][second[1]]) * 0.5 - origin
                        tip = (uv[s][first[3]] + uv[s][second[3]]) * 0.5 - origin
                        for alpha in np.linspace(0, 1.4, 40):
                            cx, cy = np.rint(base + alpha * (tip - base)).astype(int)
                            if (
                                0 <= cx < shape[1]
                                and 0 <= cy < shape[0]
                                and away[cy, cx] > safe_gap
                                and not palm_region[cy, cx]
                            ):
                                radius = max(
                                    1, min(3, int((away[cy, cx] - safe_gap) * 0.4))
                                )
                                cv2.circle(labels, (cx, cy), radius, cv2.GC_BGD, -1)

                # Bound forearm growth beyond the observed wrist band. Its
                # artificial proximal cut is excluded by preprocessing.
                if good[palm_ids].all():
                    wrist = uv[s][0]
                    direction = uv[s][palm_ids[1:]].mean(0) - wrist
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction /= norm
                        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
                        behind_wrist = (xx - wrist[0]) * direction[0] + (
                            yy - wrist[1]
                        ) * direction[1] < -max(5, 0.4 * norm)
                        labels[behind_wrist] = cv2.GC_BGD
                if color_distance is not None:
                    labels[(color_distance > 3.0) & (labels != cv2.GC_BGD)] = cv2.GC_BGD
                try:
                    # Preserve the exact graph and five iterations; only
                    # pixelwise initialization work is restricted to this ROI.
                    roi_labels = labels.copy()
                    cv2.grabCut(
                        patch,
                        roi_labels,
                        None,
                        np.zeros((1, 65)),
                        np.zeros((1, 65)),
                        5,
                        cv2.GC_INIT_WITH_MASK,
                    )
                    labels = roi_labels
                    candidate = np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(
                        np.uint8
                    )
                    if color_distance is not None:
                        candidate[(color_distance > 2.2) & (palm_region == 0)] = 0
                        # Include the original image's zero background around
                        # interior crop edges during closing, preserving border behavior.
                        top, bottom = min(2, y0), min(2, h - 1 - y1)
                        left, right = min(2, x0), min(2, w - 1 - x1)
                        padded = cv2.copyMakeBorder(
                            candidate,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=0,
                        )
                        padded = cv2.morphologyEx(
                            padded, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                        )
                        candidate = padded[top : top + shape[0], left : left + shape[1]]
                    count, components = cv2.connectedComponents(candidate)
                    if count > 1:
                        hits = np.bincount(components[seeds > 0], minlength=count)
                        hits[0] = 0
                        candidate = (components == int(np.argmax(hits))).astype(
                            np.uint8
                        )
                    result[s][roi] = candidate
                    scores[s] = mask_score(result[s], uv[s], confidence[s]) * 0.5
                except cv2.error:
                    continue
            return result, scores
        from PIL import Image
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = self.processor(
                images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                return_tensors="pt",
                device=self.device,
            ).to(self.device)
            output = self.model(
                pixel_values=inputs.pixel_values,
                text_embeds=self.text_embeds,
                attention_mask=self.attention_mask,
            )
            masks, raw_scores = _sam3_masks(output, (h, w))
        if len(masks) == 0:
            return result, scores
        association = np.array(
            [[mask_score(m, uv[s], confidence[s]) for m in masks] for s in range(2)]
        )
        for s, j in zip(*linear_sum_assignment(-association)):
            if association[s, j] >= 0.25:
                result[s] = masks[j]
                scores[s] = float(association[s, j] * raw_scores[j])
        return result, scores


def _sam3_masks(output, size):
    """Return uint8 masks (N, height, width) and scores for one image.

    ``size`` is (height, width). Preserve the original SAM3 adapter's detection
    threshold and resize logits before applying the mask threshold.
    """
    import torch
    import torch.nn.functional as F

    scores = output.pred_logits[0].sigmoid() * output.presence_logits[0].sigmoid()
    keep = scores > 0.5
    logits = output.pred_masks[0, keep]
    if len(logits) == 0:
        return np.zeros((0, *size), np.uint8), np.zeros(0, np.float32)
    # Resize logits before sigmoid, matching the previous image adapter.
    # Transformers' default postprocessor resizes probabilities instead, which
    # shifts silhouette boundaries. Keep masks compact before the CPU transfer.
    masks = (
        F.interpolate(
            logits[:, None], size=size, mode="bilinear", align_corners=False
        ).sigmoid()[:, 0]
        > 0.5
    )
    return (
        masks.to(torch.uint8).cpu().numpy(),
        scores[keep].float().cpu().numpy(),
    )

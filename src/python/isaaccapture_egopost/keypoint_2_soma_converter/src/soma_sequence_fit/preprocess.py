# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Sequential video preprocessing and compact per-frame observation storage."""

from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm
from .data import video_frames, sha256_file, save_json
from .geometry import project_np, crop_camera, FINGERS
from .segmentation import MaskProvider
from .losses import target_distance


PREPROCESSING_VERSION = 5


def preprocessing_signature():
    """Invalidate cached masks when their generating code changes."""
    return {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("preprocess.py", "segmentation.py", "geometry.py")
    }


# Frame selection and visibility scoring.


def descriptor(points):
    """Keep camera-relative palm orientation while removing size and translation."""
    p = np.asarray(points)
    if not np.isfinite(p[[0, 5, 9, 17]]).all():
        return None
    forward = p[9] - p[0]
    across = p[5] - p[17]
    normal = np.cross(across, forward)
    if np.linalg.norm(normal) < 1e-9 or np.linalg.norm(forward) < 1e-6:
        return None
    normal /= np.linalg.norm(normal)
    forward /= np.linalg.norm(forward)
    values = [*normal, *forward]
    for finger in FINGERS:
        q = p[finger]
        if not np.isfinite(q).all():
            return None
        d = np.diff(q, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
        values.extend((d[:-1] * d[1:]).sum(1).tolist())
    span = max(np.linalg.norm(p[5] - p[17]), 1e-6)
    values.extend(
        (np.linalg.norm(np.diff(p[[4, 8, 12, 16, 20]], axis=0), axis=1) / span)
        .clip(0, 3)
        .tolist()
    )
    return np.asarray(values, np.float32)


def select_diverse(descriptors, quality, count, eligible=None):
    quality = np.asarray(quality, float)
    ok = np.isfinite(descriptors).all(1) & (quality > 0)
    if eligible is not None:
        ok &= eligible
    ids = np.flatnonzero(ok)
    if not len(ids):
        return np.array([], dtype=int)
    x = descriptors[ids].astype(float)
    scale = np.maximum(np.std(x, axis=0), 0.2)
    x /= scale
    q = quality[ids] / max(quality[ids].max(), 1e-8)
    chosen = [int(np.argmax(q))]
    nearest = np.full(len(ids), np.inf)
    for _ in range(min(count, len(ids)) - 1):
        distance = np.sqrt(np.mean((x - x[chosen[-1]]) ** 2, axis=1))
        nearest = np.minimum(nearest, distance)
        score = nearest * np.sqrt(q)
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return np.sort(ids[chosen])


def surface_visibility(points, confidence, K, mask):
    """Score image evidence for hand shape, without using a fitted surface.

    Distinct projected fingers supply width evidence. Overlapping or unsupported
    shafts provide weaker evidence even if the input 3D points are confident.
    """
    uv = project_np(points, K)
    good = np.isfinite(uv).all(-1) & (confidence > 0.2)
    if not good[[5, 17]].all():
        return 0.0
    width = max(2, round(np.linalg.norm(uv[5] - uv[17]) * 0.18))
    union = np.zeros(mask.shape, np.uint8)
    area = 0.0
    support = []
    for finger in FINGERS:
        shaft = np.zeros(mask.shape, np.uint8)
        for a, b in zip(finger[:-1], finger[1:]):
            if good[a] and good[b]:
                cv2.line(
                    shaft,
                    tuple(np.rint(uv[a]).astype(int)),
                    tuple(np.rint(uv[b]).astype(int)),
                    1,
                    width,
                )
        mass = float(shaft.sum())
        if mass < 3:
            return 0.0
        area += mass
        union |= shaft
        support.append(float((shaft * mask).sum()) / mass)
    separate = float(union.sum()) / max(area, 1)
    return separate**3 * float(np.mean(support)) ** 2


def surface_training_pool(eligible, quality, visibility, fraction=0.8):
    """Reserve shared surface fitting for the clearest available hand views."""
    eligible = np.asarray(eligible, bool).copy()
    visibility = np.asarray(visibility, float)
    informative = eligible & np.isfinite(visibility) & (visibility > 0.05)
    if fraction <= 0 or informative.sum() < 4:
        return eligible, np.asarray(quality).copy()
    threshold = fraction * visibility[informative].max()
    pool = informative & (visibility >= threshold)
    if pool.sum() < 4:
        order = np.flatnonzero(informative)[np.argsort(visibility[informative])]
        pool[order[-4:]] = True
    weighted = np.asarray(quality) * np.clip(np.nan_to_num(visibility), 0.1, 1)
    return pool, weighted


def finger_balance_map(uv, confidence, mask, trust):
    """Give visible distal fingers comparable image-loss mass to the palm.

    Trust remains a separate [0,1] mask. Empty/occluded finger windows cannot
    receive an arbitrarily large gain, and no fitted geometry defines targets.
    """
    good = np.isfinite(uv).all(-1) & (confidence > 0.2)
    total = float((mask * trust).sum())
    if total < 12:
        return np.ones_like(mask, dtype=np.float32)
    palm_width = np.linalg.norm(uv[5] - uv[17]) if good[[5, 17]].all() else 15.0
    thickness = max(5, round(palm_width * 0.24))
    regions = []
    for finger in FINGERS:
        region = np.zeros(mask.shape, np.uint8)
        for a, b in zip(finger[:-1], finger[1:]):
            if good[a] and good[b]:
                cv2.line(
                    region,
                    tuple(np.rint(uv[a]).astype(int)),
                    tuple(np.rint(uv[b]).astype(int)),
                    1,
                    thickness,
                )
        mass = float((region * mask * trust).sum())
        if mass >= max(3, 0.01 * total):
            regions.append(region.astype(np.float32) / mass)
    if not regions:
        return np.ones_like(mask, dtype=np.float32)
    return np.clip(1 + total * np.mean(regions, axis=0), 1, 6).astype(np.float32)


def prepare(data, directory, args):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    provider = MaskProvider(
        args.mask_backend, args.mask_dir, args.device, args.sam3_checkpoint
    )
    quality = np.zeros((data.nframes, 2), np.float32)
    visibility = np.zeros((data.nframes, 2), np.float32)
    descriptors = np.full((data.nframes, 2, 20), np.nan, np.float32)
    usable = np.zeros((data.nframes, 2), bool)
    source_y, source_x = np.indices((data.height, data.width), dtype=np.float32)

    def prepare_frame(t, frame):
        # OpenCV's RNG is thread-local. Explicit per-frame seeds make masks
        # reproducible regardless of worker scheduling or preceding frames.
        if args.mask_backend == "grabcut":
            cv2.setRNGSeed((args.seed + int(t)) % 2147483647)
        masks, scores = provider(
            frame, t, data.points[t], data.confidence[t], data.K[t]
        )
        overlap = ((masks[0] > 0) & (masks[1] > 0)).astype(np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for s, side in enumerate(("left", "right")):
            uv = project_np(data.points[t, s], data.K[t])
            good = np.isfinite(uv).all(1) & (data.confidence[t, s] > 0.2)
            if (
                good.sum() < 6
                or not np.isfinite(data.points[t, s, [0, 5, 9, 17]]).all()
            ):
                continue
            lo, hi = uv[good].min(0), uv[good].max(0)
            span = max(float((hi - lo).max()), 8)
            center = (lo + hi) / 2
            half = 0.7 * span
            box = np.array(
                [
                    center[0] - half,
                    center[1] - half,
                    center[0] + half,
                    center[1] + half,
                ],
                np.float32,
            )
            Kcrop, A = crop_camera(data.K[t], box, args.image_size)
            mask = cv2.warpAffine(
                masks[s].astype(np.float32),
                A[:2],
                (args.image_size, args.image_size),
                flags=cv2.INTER_LINEAR,
            )
            size = (args.image_size, args.image_size)
            # Sample source pixel coordinates with exactly the same nearest
            # interpolation as the previous full-frame trust-mask warp.
            xx = cv2.warpAffine(
                source_x, A[:2], size, flags=cv2.INTER_NEAREST, borderValue=-1
            ).astype(np.float64)
            yy = cv2.warpAffine(
                source_y, A[:2], size, flags=cv2.INTER_NEAREST, borderValue=-1
            ).astype(np.float64)
            weights = ((xx >= 0) & (yy >= 0)).astype(np.float32)
            # Observe the proximal palm and lateral wrist. Only the uncertain
            # forearm cut beyond this band is ignored; excluding everything
            # behind +10% palm length concealed undersized native surfaces.
            wrist = uv[0]
            palm = np.nanmean(uv[[5, 9, 13, 17]], axis=0)
            direction = palm - wrist
            length = np.linalg.norm(direction)
            wrist_region = np.zeros(size[::-1], np.float32)
            if length > 2:
                signed = (
                    (xx - wrist[0]) * direction[0] + (yy - wrist[1]) * direction[1]
                ) / length
                weights[signed < -0.20 * length] = 0
                wrist_region = (
                    (signed >= -0.20 * length) & (signed <= 0.25 * length)
                ).astype(np.float32)
            weights[
                cv2.warpAffine(overlap, A[:2], size, flags=cv2.INTER_NEAREST) > 0
            ] = 0
            if args.ignore_dir:
                path = Path(args.ignore_dir) / side / f"{t:06d}.png"
                if path.exists():
                    ignore = cv2.imread(str(path), 0)
                    if ignore is None or ignore.shape != (data.height, data.width):
                        raise ValueError(f"Invalid ignore mask: {path}")
                    weights[
                        cv2.warpAffine(ignore, A[:2], size, flags=cv2.INTER_NEAREST)
                        > 127
                    ] = 0
            weights[:2] = 0
            weights[-2:] = 0
            weights[:, :2] = 0
            weights[:, -2:] = 0
            trusted_area = float((mask * weights).sum())
            mask_ok = trusted_area >= 12 and scores[s] >= 0.15
            if not mask_ok:
                weights[:] = 0
            patch = cv2.warpAffine(gray, A[:2], (args.image_size, args.image_size))
            sharpness = float(cv2.Laplacian(patch, cv2.CV_32F).var())
            in_frame = (
                (uv[good, 0] > 2)
                & (uv[good, 0] < data.width - 3)
                & (uv[good, 1] > 2)
                & (uv[good, 1] < data.height - 3)
            ).mean()
            q = (
                scores[s]
                * min(span / 100, 1)
                * min(max(sharpness, 0) / 100, 1)
                * float(in_frame)
            )
            visibility[t, s] = (
                surface_visibility(
                    data.points[t, s], data.confidence[t, s], Kcrop, mask
                )
                if mask_ok
                else 0
            )
            desc = descriptor(data.points[t, s])
            if desc is not None:
                descriptors[t, s] = desc
            quality[t, s] = max(q, 0.01) if mask_ok else 0
            usable[t, s] = True
            side_dir = directory / side
            side_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                side_dir / f"{t:06d}.npz",
                mask=mask.astype(np.float32),
                weight=weights,
                wrist_region=wrist_region,
                distance=target_distance(mask, weights),
                balance=finger_balance_map(
                    project_np(data.points[t, s], Kcrop),
                    data.confidence[t, s],
                    mask,
                    weights,
                ),
                K=Kcrop,
                box=box,
                mask_valid=np.array(mask_ok),
            )

    workers = (
        args.preprocess_workers if args.mask_backend in {"grabcut", "files"} else 1
    )
    # Small ROI operations should not each start a large OpenCV thread team.
    # Parallelize independent frames with a bounded queue instead.
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with tqdm(total=data.nframes, desc="Segment and prepare") as progress:
            if workers == 1:
                for t, frame in video_frames(data):
                    prepare_frame(t, frame)
                    progress.update()
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    pending = deque()
                    for t, frame in video_frames(data):
                        pending.append(executor.submit(prepare_frame, t, frame))
                        if len(pending) >= 2 * workers:
                            pending.popleft().result()
                            progress.update()
                    while pending:
                        pending.popleft().result()
                        progress.update()
    finally:
        cv2.setNumThreads(previous_threads)
    # Reject severe isolated landmark size jumps without treating all fast motion as failure.
    for s in range(2):
        lengths = np.linalg.norm(data.points[:, s, 5] - data.points[:, s, 17], axis=-1)
        for t in np.flatnonzero(quality[:, s] > 0):
            neighbors = lengths[max(0, t - 2) : min(data.nframes, t + 3)]
            ref = np.nanmedian(neighbors)
            if np.isfinite(ref) and ref > 1e-5 and not 0.6 < lengths[t] / ref < 1.6:
                quality[t, s] *= 0.1
    np.savez_compressed(
        directory / "index.npz",
        quality=quality,
        descriptors=descriptors,
        usable=usable,
        surface_visibility=visibility,
    )
    save_json(
        directory / "provenance.json",
        {
            "video_sha256": sha256_file(data.video),
            "points_sha256": sha256_file(data.points_file),
            "mask_backend": args.mask_backend,
            "mask_seed": args.seed,
            "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessing_source_sha256": preprocessing_signature(),
            "image_size": args.image_size,
            "frame_count": data.nframes,
        },
    )
    return quality, descriptors, usable


class ObservationStore:
    def __init__(
        self, directory, side, device, host_cache_mib=256, device_cache_mib=128
    ):
        self.directory = Path(directory) / side
        self.device = device
        self.cache = {}
        self.cache_bytes = 0
        if host_cache_mib <= 0:
            raise ValueError("host_cache_mib must be positive")
        self.cache_budget_bytes = int(host_cache_mib * 1024 * 1024)
        self.device_batches = {}
        self.device_bytes = 0
        if device_cache_mib < 0:
            raise ValueError("device_cache_mib must be nonnegative")
        self.device_budget_bytes = int(device_cache_mib * 1024 * 1024)

    def load(self, ids):
        import torch

        key = tuple(int(i) for i in ids)
        if key in self.device_batches:
            batch = self.device_batches.pop(key)
            self.device_batches[key] = batch
            return batch
        rows = []
        for i in ids:
            i = int(i)
            if i not in self.cache:
                with np.load(self.directory / f"{i:06d}.npz", allow_pickle=False) as d:
                    self.cache[i] = {
                        k: np.array(d[k]) for k in ("mask", "weight", "distance", "K")
                    }
                    self.cache[i]["wrist_region"] = (
                        np.array(d["wrist_region"])
                        if "wrist_region" in d
                        else np.zeros_like(d["mask"])
                    )
                    self.cache[i]["balance"] = (
                        np.array(d["balance"])
                        if "balance" in d
                        else np.ones_like(d["mask"])
                    )
                self.cache_bytes += sum(
                    value.nbytes for value in self.cache[i].values()
                )
                while (
                    self.cache_bytes > self.cache_budget_bytes and len(self.cache) > 1
                ):
                    old = self.cache.pop(next(iter(self.cache)))
                    self.cache_bytes -= sum(value.nbytes for value in old.values())
            else:
                # LRU by access, bounded by bytes rather than a short frame count.
                self.cache[i] = self.cache.pop(i)
            rows.append(self.cache[i])
        batch = {
            k: torch.tensor(
                np.stack([r[k] for r in rows]), device=self.device, dtype=torch.float32
            )
            for k in rows[0]
        }
        # Renderer chunks repeat on every sequence iteration. Keep their
        # immutable tensors on device, instead of restacking/copying each time.
        if str(self.device).startswith("cuda"):
            size = sum(value.numel() * value.element_size() for value in batch.values())
            if size <= self.device_budget_bytes:
                while (
                    self.device_batches
                    and self.device_bytes + size > self.device_budget_bytes
                ):
                    old = self.device_batches.pop(next(iter(self.device_batches)))
                    self.device_bytes -= sum(
                        value.numel() * value.element_size() for value in old.values()
                    )
                self.device_batches[key] = batch
                self.device_bytes += size
        return batch

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Temporal identity assignment + anchor-freeze box rules for RT-DETR hand tracks.

Vendored from the standalone ``hand_detector`` package
(``hand_detector/hand_tracker.py``). The upstream ``track_video`` helper is
dropped -- EgoHand feeds already-decoded frames -- but the algorithm and its tuned
constants are unchanged.

Why this exists: ``onnx_model.HandDetector`` decides every frame independently.
On video that drops boxes on marginal frames and swaps left/right identity,
because the detector's per-frame left/right classification is not reliable -- on
roughly 2% of two-hand frames the box labelled ``left_hand`` is not the leftmost
of the two. Identity is therefore resolved here from geometry and track
continuity, and the class label is used only as a way to collect candidates and
their scores.

This is a WHOLE-SEQUENCE algorithm, not streaming: ``gap_fill`` interpolates
between the previous and the NEXT real detection, so it needs the full clip.

Gap filling produces boxes throughout the clip for hands detected at least
once. Left/right screen ordering is optionally enforced.
"""

import numpy as np

__all__ = [
    "CONSTANTS",
    "HandTracker",
    "apply_bbox_rules",
    "candidate_label",
    "select_new",
]

THR = 0.15  # a candidate above this score is usable on its own
THR_RESCUE = 0.15  # a sub-threshold candidate may still rescue a track
RESCUE_IOU = 0.35  # ...if it overlaps the previous box at least this much
W_DET, W_IOU = 1.0, 1.5  # association weights: continuity outranks a small score edge
PAD = 1.4  # direct side multiplier, i.e. VitPose-convention padding=0.4
MIN_SIZE_FRAC = 0.16
MAX_CENTER_SPEED = 90.0
MAX_SIZE_RATIO = 1.12
CENTER_STILL = 8.0

CONSTANTS = dict(
    THR=THR,
    THR_RESCUE=THR_RESCUE,
    RESCUE_IOU=RESCUE_IOU,
    W_DET=W_DET,
    W_IOU=W_IOU,
    PAD=PAD,
    MIN_SIZE_FRAC=MIN_SIZE_FRAC,
    MAX_CENTER_SPEED=MAX_CENTER_SPEED,
    MAX_SIZE_RATIO=MAX_SIZE_RATIO,
    CENTER_STILL=CENTER_STILL,
)


def iou_xywh(a, b) -> float:
    """IoU of two centre-form boxes [cx,cy,w,h] (or [cx,cy,s] treated as square)."""
    if a is None or b is None:
        return 0.0
    aw, ah = (a[2], a[3]) if len(a) > 3 else (a[2], a[2])
    bw, bh = (b[2], b[3]) if len(b) > 3 else (b[2], b[2])
    ax0, ay0, ax1, ay1 = a[0] - aw / 2, a[1] - ah / 2, a[0] + aw / 2, a[1] + ah / 2
    bx0, by0, bx1, by1 = b[0] - bw / 2, b[1] - bh / 2, b[0] + bw / 2, b[1] + bh / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def to_square(c):
    """candidate [cx,cy,w,h,score] -> padded square box [cx,cy,size]."""
    return [c[0], c[1], max(c[2], c[3]) * PAD]


def candidate_label(c):
    """Class id carried on a candidate, or ``None`` for a label-less record.

    Candidates are ``[cx, cy, w, h, score]`` upstream and ``[..., label_id]``
    here; callers that build records by hand may still omit the label.
    """
    return c[5] if len(c) > 5 else None


def pool_candidates(rec, prevL, prevR):
    """All candidates from BOTH classes, de-duplicated.

    The class label is ignored for association -- that is the whole point of the
    2x2 assignment in ``select_new`` -- but it is carried through at index 5 so
    the bootstrap frame can fall back on it (see ``select_new``).
    """
    pool: list[list] = []
    for key in ("L", "R"):
        for c in rec[key]:
            dup = None
            for p in pool:
                if iou_xywh(p, c) > 0.9:
                    dup = p
                    break
            if dup is None:
                pool.append(list(c))
            elif c[4] > dup[4]:
                # Same box under both labels -> keep the better score, and the
                # label that came with it.
                dup[4] = c[4]
                if len(dup) > 5 and len(c) > 5:
                    dup[5] = c[5]
    usable = []
    for c in pool:
        if c[4] > THR:
            usable.append(c)
        elif c[4] >= THR_RESCUE and (
            iou_xywh(c, prevL) >= RESCUE_IOU or iou_xywh(c, prevR) >= RESCUE_IOU
        ):
            usable.append(c)  # rescue: continues a track, beats a synthetic gap-fill
    usable.sort(key=lambda c: -c[4])
    return usable[:5]


def select_new(det):
    """Pooled candidates + 2x2 identity assignment. Returns (tracks, stats).

    Each frame solves a tiny assignment problem: give the left and right TRACK the
    pair of candidates maximising sum(W_DET*score + W_IOU*IoU(candidate, that
    track's last box)). A candidate can be used at most once, so mutual exclusion
    is structural. Identity comes from overlap rather than x-order, so it survives
    a genuine hand crossing.

    DEVIATION FROM UPSTREAM: on the bootstrap frame a *lone* candidate is assigned
    by its class label rather than by x-order. x-order carries no information about
    a single box, so upstream sent every lone bootstrap candidate to the left
    track: a clip showing only the right hand emitted right-hand boxes under
    ``left``, and IoU continuity kept them there for the rest of the clip. Two
    candidates bootstrapping together still use x-order, where it is the stronger
    signal. Records built by hand without a ``label_id`` fall back to x-order too.
    """
    out = {"L": ([], []), "R": ([], [])}
    prevL = prevR = None
    stats = {"rescued": 0, "excluded": 0, "label_ignored": 0, "crossed": 0}
    for rec in det:
        usable = pool_candidates(rec, prevL, prevR)
        L = R = None
        if not usable:
            pass
        elif prevL is None and prevR is None:
            if len(usable) == 1 and candidate_label(usable[0]) is not None:
                if candidate_label(usable[0]) == 1:
                    L = usable[0]
                else:
                    R = usable[0]
            else:
                s = sorted(usable[:2], key=lambda c: c[0])
                L = s[0]
                R = s[1] if len(s) > 1 else None
        else:
            best = None
            opts = usable + [None]
            for a in opts:
                for b in opts:
                    if a is not None and b is not None and a is b:
                        continue
                    val = 0.0
                    if a is not None:
                        val += W_DET * a[4] + W_IOU * iou_xywh(a, prevL)
                    if b is not None:
                        val += W_DET * b[4] + W_IOU * iou_xywh(b, prevR)
                    if a is None and b is None:
                        val = -1e9
                    if best is None or val > best[0]:
                        best = (val, a, b)
            L, R = best[1], best[2]
        topL = rec["L"][0] if rec["L"] else None
        if L is not None and topL is not None and iou_xywh(L, topL) < 0.5:
            stats["label_ignored"] += 1
        if L is not None and R is not None and L[0] > R[0]:
            stats["crossed"] += 1
        for key, c in (("L", L), ("R", R)):
            if c is None:
                out[key][0].append([0.0, 0.0, 0.0])
                out[key][1].append(False)
            else:
                if c[4] <= THR:
                    stats["rescued"] += 1
                out[key][0].append(to_square(c))
                out[key][1].append(True)
        if L is not None:
            prevL = L
        if R is not None:
            prevR = R
    return (
        {k: (np.array(v[0], np.float32), np.array(v[1], bool)) for k, v in out.items()},
        stats,
    )


def enforce_lr_per_frame(tracks):
    """Per frame, the smaller-x box IS the left hand; swap where that is violated.

    DEVIATION FROM UPSTREAM: the swap only fires when BOTH hands are present on
    the frame. ``select_new`` writes a ``[0.0, 0.0, 0.0]`` sentinel for an absent
    track, and ``cx > 0.0`` is true for essentially every real box, so the
    unguarded upstream comparison swapped every *singleton* frame whose detection
    happened to land on the left track -- systematically draining the left track
    onto the right one and leaving the left hand to be gap-filled from
    interpolated garbage. Comparing a real box against a sentinel is meaningless;
    only a genuine both-present pair carries screen-order information.
    """
    if "left" not in tracks or "right" not in tracks:
        return tracks
    lb, lp = tracks["left"]
    rb, rp = tracks["right"]
    n = min(len(lb), len(rb))
    for t in range(n):
        if lp[t] and rp[t] and lb[t, 0] > rb[t, 0]:
            lb[t], rb[t] = rb[t].copy(), lb[t].copy()
            lp[t], rp[t] = rp[t], lp[t]
    return tracks


def apply_bbox_rules(
    bbx,
    present,
    min_size,
    max_center_speed,
    max_size_ratio,
    center_still,
    still_mode="anchor",
):
    """Robustness rules on one hand's [cx,cy,size] track. Returns (bbx, present).

    Rules, in order:
      1. gap_fill    - missing boxes linearly interpolated from the previous AND next
                       real detection; leading/trailing gaps hold the nearest one.
      2. min_size    - clamp the side length up to ``min_size``.
      3. size_freeze - if the hand is ~stationary the size shouldn't change.
                       still_mode="anchor": displacement from an ANCHOR centre <=
                       center_still, the anchor re-planted whenever the box escapes
                       that radius, so slow drift eventually unfreezes the size.
                       still_mode="frame" instead tests the per-frame step, which
                       lets a hand creeping below the threshold stay frozen
                       indefinitely; "anchor" is the sane default.
                       The anchor test uses DISPLACEMENT, not path length.
      4. size_rate   - otherwise limit per-frame size change to max_size_ratio.
      5. center_speed- limit per-frame centre movement to max_center_speed px.
    """
    bbx = bbx.astype(np.float64).copy()
    present = present.copy()
    N = len(bbx)

    idx = np.where(present)[0]
    if len(idx) == 0:
        return bbx.astype(np.float32), np.ones(N, bool)
    # 1) bidirectional gap fill
    for t in range(idx[0]):
        bbx[t] = bbx[idx[0]]
        present[t] = True
    for t in range(idx[-1] + 1, N):
        bbx[t] = bbx[idx[-1]]
        present[t] = True
    for a, b in zip(idx, idx[1:]):
        if b - a > 1:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                bbx[t] = (1 - w) * bbx[a] + w * bbx[b]
                present[t] = True
    # 2) minimum size
    for t in range(N):
        if bbx[t, 2] < min_size:
            bbx[t, 2] = min_size
    # 3-5) forward pass against the previous (stabilised) box
    anchor_c = bbx[0, :2].copy()
    for t in range(1, N):
        prev_c = bbx[t - 1, :2].copy()
        s_prev = bbx[t - 1, 2]
        d = bbx[t, :2] - prev_c
        dist = float(np.hypot(d[0], d[1]))
        if still_mode == "anchor":
            ad = bbx[t, :2] - anchor_c
            still = float(np.hypot(ad[0], ad[1])) <= center_still
            if not still:
                anchor_c = bbx[t, :2].copy()
        else:
            still = dist <= center_still
        if still:
            if s_prev > 0 and abs(bbx[t, 2] - s_prev) > 1e-6:
                bbx[t, 2] = s_prev
        elif s_prev > 0:
            hi, lo = s_prev * max_size_ratio, s_prev / max_size_ratio
            if bbx[t, 2] > hi:
                bbx[t, 2] = hi
            elif bbx[t, 2] < lo:
                bbx[t, 2] = lo
        if dist > max_center_speed:
            bbx[t, :2] = prev_c + d * (max_center_speed / dist)

    return bbx.astype(np.float32), present


class HandTracker:
    """Whole-sequence hand track builder.

    Mutual exclusion between the two tracks needs no IoU threshold, because
    the 2x2 assignment in ``select_new`` already uses each candidate at
    most once.

    Args:
        image_width: full-image width in pixels; sets ``MIN_SIZE_FRAC * width``.
        still_mode: ``"anchor"`` (default) or ``"frame"``; see ``apply_bbox_rules``.
        enforce_lr_order: when true (the upstream default) the smaller-x box is
            forced to be the left hand on every frame. This is tuned for
            egocentric footage where the wearer's hands rarely cross; set it
            false to keep the geometric identity assignment from ``select_new``,
            which survives a genuine crossing.
    """

    def __init__(
        self,
        image_width: int,
        still_mode: str = "anchor",
        enforce_lr_order: bool = True,
    ):
        self.min_size = MIN_SIZE_FRAC * image_width
        self.still_mode = still_mode
        self.enforce_lr_order = enforce_lr_order

    @staticmethod
    def detections_to_records(per_frame) -> list[dict]:
        """[[Detection, ...], ...] -> [{'L': [[cx,cy,w,h,score,label_id],...], 'R': [...]}, ...]"""
        recs = []
        for dets in per_frame:
            rec = {"L": [], "R": []}
            for d in sorted(dets, key=lambda d: -d.score):
                x1, y1, x2, y2 = d.xyxy
                cand = [
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1,
                    d.score,
                    d.label_id,
                ]
                rec["L" if d.label_id == 1 else "R"].append(cand)
            recs.append(rec)
        return recs

    def track(self, per_frame) -> dict:
        """Build stabilised left/right tracks.

        Returns ``{'left': (N,3), 'right': (N,3)}`` of ``[cx, cy, size]``, plus
        ``'stats'`` and ``'raw_present'``. A box is present on every frame after
        gap_fill, so the returned tracks have no gaps.

        DEVIATION FROM UPSTREAM: ``raw_present`` -- the pre-gap-fill presence mask
        -- is returned as well. ``apply_bbox_rules`` short-circuits to all-zero
        boxes flagged present when a hand is never detected, so the adapter needs
        the mask to emit nothing for that hand rather than a ``size=0`` box that
        would divide by zero in ``crop_and_preprocess``. It also backs
        ``drop_gap_filled`` and the per-hand coverage logging.
        """
        recs = (
            per_frame
            if per_frame and isinstance(per_frame[0], dict)
            else self.detections_to_records(per_frame)
        )
        sel, stats = select_new(recs)

        # Order matters: settle left/right BEFORE the rules, never after. A label
        # flip otherwise puts an ~800px jump inside the track, center_speed clamps
        # it to 90px/frame, and the box crawls across the frame for ~9 frames. The
        # post-rules pass is deliberately omitted -- rerunning it would re-break a
        # genuine hand crossing.
        tr = {side: sel[key] for side, key in (("left", "L"), ("right", "R"))}
        if self.enforce_lr_order:
            tr = enforce_lr_per_frame(tr)

        out = {"stats": stats, "raw_present": {}}
        for side in ("left", "right"):
            b, p = tr[side]
            out["raw_present"][side] = p.copy()
            f, _ = apply_bbox_rules(
                b.copy(),
                p.copy(),
                self.min_size,
                MAX_CENTER_SPEED,
                MAX_SIZE_RATIO,
                CENTER_STILL,
                still_mode=self.still_mode,
            )
            out[side] = f
        return out

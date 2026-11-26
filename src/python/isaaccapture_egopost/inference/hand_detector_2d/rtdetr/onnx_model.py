# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""RT-DETRv2 R18 egocentric hand detector, ONNX runtime only.

Vendored from the standalone ``hand_detector`` package
(``hand_detector/hand_detector.py``). Only the library
surface is kept: the CLI, the video reader, and the PIL drawing helpers are
dropped because EgoHand feeds already-decoded frames and renders overlays itself.

No PyTorch and no RT-DETR source tree are required -- just ``onnxruntime`` +
``numpy`` + ``Pillow``. ``onnxruntime-gpu`` is pinned in requirements.txt, but this module is still
imported lazily (see ``detector.py``) so a checkout without it -- a bare dev env,
or an image built before the pin landed -- can still import the rest of the
package and run with ground-truth or saved boxes.

Class ids come from the training config (``num_classes: 3``, index 0 unused):
1 = ``left_hand``, 2 = ``right_hand``, anatomical to the *wearer* rather than
left/right position in the frame.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

__all__ = ["Detection", "HandDetector", "LABELS", "refine_box"]

LABELS = {1: "left_hand", 2: "right_hand"}
# Maps the ONNX class id onto the hand-side strings EgoHand uses everywhere else.
LABEL_ID_TO_HAND = {1: "left", 2: "right"}
INPUT_SIZE = 640  # the ONNX graph was traced at 640x640


#: Globs under ``site-packages/nvidia/`` holding the shared objects ORT needs,
#: in load order -- cuBLAS first, because cuDNN resolves against its symbols.
#:
#: The CUDA 12 and CUDA 13 wheels do not agree on layout, and getting this wrong
#: costs an 8x silent slowdown rather than an error:
#:
#:   nvidia-cublas-cu12  -> nvidia/cublas/lib/libcublas.so.12
#:   nvidia-cudnn-cu12   -> nvidia/cudnn/lib/libcudnn.so.9
#:   nvidia-cublas       -> nvidia/cu13/lib/libcublas.so.13   <-- NOT nvidia/cublas
#:   nvidia-cudnn-cu13   -> nvidia/cudnn/lib/libcudnn.so.9
#:
#: (``nvidia-cublas-cu13`` on PyPI is an unrelated 0.0.1 stub; the real CUDA 13
#: cuBLAS wheel is the unsuffixed ``nvidia-cublas``.)
_CUDA_LIB_GLOBS = (
    "cublas/lib/lib*.so*",  # CUDA 12
    "cu13/lib/lib*.so*",  # CUDA 13 (cuBLAS and friends share one directory)
    "cudnn/lib/lib*.so*",  # both
)


def _preload_cuda_libs() -> None:
    """dlopen the pip-installed NVIDIA libs so ORT's CUDA provider can find them.

    onnxruntime-gpu links against libcudnn/libcublas by soname and relies on the
    dynamic loader. When CUDA came from pip wheels (nvidia-cudnn-cu12 etc., which
    is what ``pip install torch`` pulls in) those live under site-packages/nvidia/
    and are not on the default search path, so the CUDA provider silently falls
    back to CPU. Loading them here with RTLD_GLOBAL puts them in the process's
    symbol table before ORT looks. No-op if the wheels aren't installed, which is
    the normal case in an image whose base already ships CUDA system-wide.
    """
    import ctypes
    import glob
    import site

    roots = [Path(p) / "nvidia" for p in site.getsitepackages()]
    roots += [Path(site.getusersitepackages()) / "nvidia"]
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in _CUDA_LIB_GLOBS:
            for so in sorted(glob.glob(str(root / pattern))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError as e:
                    # Best effort: ORT falls back to the system loader path.
                    log.debug("Could not preload %s: %s", so, e)
        break


@dataclass
class Detection:
    label: str
    label_id: int
    score: float
    xyxy: tuple[float, float, float, float]

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """COCO order: (x, y, width, height)."""
        x1, y1, x2, y2 = self.xyxy
        return (x1, y1, x2 - x1, y2 - y1)

    @property
    def cxcys(self) -> tuple[float, float, float]:
        """(centre_x, centre_y, size) -- the square-crop convention EgoHand consumes.

        ``size`` is the longer side, so this is only a faithful description of
        the box when the box is already square (``square=True``).
        """
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["xyxy"] = [round(v, 2) for v in self.xyxy]
        d["score"] = round(self.score, 4)
        return d


def refine_box(
    xyxy,
    img_w: float,
    img_h: float,
    pad: float = 0.0,
    square: bool = False,
    min_size: float = 0.0,
    clip: bool = True,
):
    """Expand / squarify / clamp one box. Returns a new (x1, y1, x2, y2).

    Order is deliberate: pad -> square -> min_size -> clip. Squaring after
    padding keeps the pad symmetric; clipping last guarantees the final box is
    inside the image whatever the earlier steps did.

    Note the ``pad`` convention: each side grows by ``pad`` of the box, so the
    side length is multiplied by ``1 + 2 * pad``. The EgoHand convention
    elsewhere instead multiplies the side by ``1 + padding``; ``detector.py``
    converts between them.
    """
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    if pad:
        w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
    if square:
        w = h = max(w, h)
    if min_size:
        w, h = max(w, min_size), max(h, min_size)

    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    if clip:
        # Shift the box back inside before resorting to shrinking it -- a shift
        # preserves the size (and squareness) that padding just established,
        # which matters when the crop feeds a pose model expecting a fixed
        # aspect. Only a box larger than the image itself gets shrunk.
        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if x2 > img_w:
            x1 -= x2 - img_w
            x2 = float(img_w)
        if y2 > img_h:
            y1 -= y2 - img_h
            y2 = float(img_h)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(img_w), x2), min(float(img_h), y2)

    return (x1, y1, x2, y2)


class HandDetector:
    """RT-DETRv2 hand detector.

    Args:
        weights: path to the .onnx file.
        score_thr: minimum confidence to return.
        providers: onnxruntime execution providers. ``None`` selects CUDA if
            available, else CPU.
        top1_per_class: keep only the highest-scoring box per hand.
        pad: expand each side by this fraction of the box (side x (1 + 2 * pad)).
        square: expand the shorter side so the box is square.
        min_size: enforce a minimum box side in pixels.
        clip: keep boxes inside the image.
    """

    def __init__(
        self,
        weights: str | Path,
        score_thr: float = 0.5,
        providers: list | None = None,
        top1_per_class: bool = False,
        pad: float = 0.0,
        square: bool = False,
        min_size: float = 0.0,
        clip: bool = True,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "The rtdetr hand detector needs onnxruntime-gpu, which requirements.txt "
                "pins alongside nvidia-cudnn-cu13. Install them with "
                "`pip install -r requirements.txt`, or pick a build matching your CUDA "
                "major (<=1.26 is CUDA 12, >=1.27 is CUDA 13), or use "
                "--bbox_detector_config .../saved_bboxes.yaml with your own boxes."
            ) from exc

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"RT-DETR hand detector weights not found: {weights}"
            )

        if providers is None:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                _preload_cuda_libs()
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        elif any(
            (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
            for p in providers
        ):
            _preload_cuda_libs()

        self.session = ort.InferenceSession(str(weights), providers=providers)
        self.score_thr = score_thr
        self.providers = self.session.get_providers()
        self.top1_per_class = top1_per_class
        self.pad = pad
        self.square = square
        self.min_size = min_size
        self.clip = clip

    # ---------------------------------------------------------------- preproc

    @staticmethod
    def _preprocess(img: Image.Image) -> np.ndarray:
        """Resize to 640x640 and scale to [0,1] CHW float32.

        Deliberately a plain resize, NOT letterboxing: the model was trained on
        squashed 640x640 crops (RT-DETR's default Resize transform), so
        preserving aspect ratio here would introduce a train/test mismatch.
        The postprocessor rescales boxes back to original size for us.
        """
        arr = (
            np.asarray(
                img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )
        return arr.transpose(2, 0, 1)[None]  # 1,3,H,W

    def _run(self, batch: np.ndarray, sizes: np.ndarray) -> list[list[Detection]]:
        labels, boxes, scores = self.session.run(
            ["labels", "boxes", "scores"],
            {"images": batch, "orig_target_sizes": sizes},
        )
        out = []
        for lab, box, sc, (iw, ih) in zip(labels, boxes, scores, sizes):
            keep = sc > self.score_thr
            dets = [
                Detection(
                    LABELS.get(int(label), str(int(label))),
                    int(label),
                    float(score),
                    tuple(float(v) for v in bbox),
                )
                for label, bbox, score in zip(lab[keep], box[keep], sc[keep])
            ]
            dets.sort(key=lambda d: -d.score)

            if self.top1_per_class:
                # dets is already score-sorted, so the first hit per class wins.
                seen, best = set(), []
                for d in dets:
                    if d.label_id not in seen:
                        seen.add(d.label_id)
                        best.append(d)
                dets = best

            if self.pad or self.square or self.min_size or self.clip:
                for d in dets:
                    d.xyxy = refine_box(
                        d.xyxy, iw, ih, self.pad, self.square, self.min_size, self.clip
                    )
            out.append(dets)
        return out

    # ------------------------------------------------------------------- api

    def detect(self, image: str | Path | Image.Image) -> list[Detection]:
        """Detect hands in one image. Returns detections sorted by score."""
        img = Image.open(image) if not isinstance(image, Image.Image) else image
        w, h = img.size
        return self._run(self._preprocess(img), np.array([[w, h]], dtype=np.int64))[0]

    def detect_batch(self, images: list) -> list[list[Detection]]:
        """Detect on several images at once.

        Images may differ in size: each is resized to 640 independently and the
        model rescales each result with its own ``orig_target_size``.
        """
        pil = [Image.open(i) if not isinstance(i, Image.Image) else i for i in images]
        batch = np.concatenate([self._preprocess(p) for p in pil], axis=0)
        sizes = np.array([[p.size[0], p.size[1]] for p in pil], dtype=np.int64)
        return self._run(batch, sizes)

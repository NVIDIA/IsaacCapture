# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Mesh-overlay rendering helpers, and render_clip, used by the EgoHand inference pipeline.

render_clip is what inference/scripts/run_egohand_inference_pipeline.py calls to
render the bbox/crop/fullimg overlay videos after EgoHand hand inference.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from egohand.utils.cam_utils import compute_transl_full_cam
from inference.visualization.drawing import draw_bbox_on_image, draw_hand_skeleton

# Lazy-import Renderer so that scripts importing demo.py for checkpoint
# utilities (e.g. evaluate.py) don't need pyrender/OpenGL installed.
Renderer = None


def _get_renderer_class():
    global Renderer
    if Renderer is None:
        from egohand.utils.renderer import Renderer as _R

        Renderer = _R
    return Renderer


IMG_SIZE = 256


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_crop_overlay(
    result,
    img_crop_rgb,
    bbx_xys,
    K_fullimg,
    faces,
    model_type,
    model_focal_length=5000.0,
    mesh_color=(0.4, 0.8, 0.4),
    renderer=None,
):
    """Render mesh overlay on 256x256 crop image.

    Returns: (256, 256, 3) uint8 RGB image
    """
    h, w = img_crop_rgb.shape[:2]
    vertices = result["pred_vertices"]
    cam_t = result["pred_cam_t"]

    if model_type == "egohand":
        bbx_size = bbx_xys[2]
        scale = w / bbx_size
        f_crop = K_fullimg[0, 0] * scale
        cx_crop = (K_fullimg[0, 2] - bbx_xys[0]) * scale + w / 2.0
        cy_crop = (K_fullimg[1, 2] - bbx_xys[1]) * scale + h / 2.0
        if renderer is None:
            renderer = _get_renderer_class()(focal_length=f_crop, faces=faces)
        else:
            renderer.focal_length = f_crop
        return renderer(
            vertices,
            cam_t,
            img_crop_rgb,
            mesh_base_color=mesh_color,
            camera_center=[cx_crop, cy_crop],
        )
    else:
        if "K" in result:
            K = result["K"]
            bbx_size = bbx_xys[2]
            scale = w / bbx_size
            f_crop = K[0, 0] * scale
            cx_crop = (K[0, 2] - bbx_xys[0]) * scale + w / 2.0
            cy_crop = (K[1, 2] - bbx_xys[1]) * scale + h / 2.0
            if renderer is None:
                renderer = _get_renderer_class()(focal_length=f_crop, faces=faces)
            else:
                renderer.focal_length = f_crop
            return renderer(
                vertices,
                cam_t,
                img_crop_rgb,
                mesh_base_color=mesh_color,
                camera_center=[cx_crop, cy_crop],
            )
        else:
            if renderer is None:
                renderer = _get_renderer_class()(
                    focal_length=model_focal_length, faces=faces
                )
            return renderer(vertices, cam_t, img_crop_rgb, mesh_base_color=mesh_color)


def draw_hand_mesh(
    result,
    full_img_rgb,
    K_fullimg,
    bbx_xys,
    faces,
    model_type,
    model_focal_length=5000.0,
    mesh_color=(0.4, 0.8, 0.4),
    renderer=None,
):
    """Render mesh overlay on full-resolution image.

    Returns: (H, W, 3) uint8 RGB image
    """
    vertices = result["pred_vertices"]
    cam_t = result["pred_cam_t"]

    if model_type == "egohand":
        # The final overlay is on the raw input video, whose camera may have fx != fy.
        # A scalar would make the renderer use fx for both axes.
        focal = (K_fullimg[0, 0], K_fullimg[1, 1])
        cx = K_fullimg[0, 2]
        cy = K_fullimg[1, 2]
        if renderer is None:
            renderer = _get_renderer_class()(focal_length=focal, faces=faces)
        else:
            renderer.focal_length = focal
        return renderer(
            vertices,
            cam_t,
            full_img_rgb,
            mesh_base_color=mesh_color,
            camera_center=[cx, cy],
        )
    else:
        if "K" in result:
            K = result["K"]
            if renderer is None:
                renderer = _get_renderer_class()(focal_length=K[0, 0], faces=faces)
            else:
                renderer.focal_length = K[0, 0]
            return renderer(
                vertices,
                cam_t,
                full_img_rgb,
                mesh_base_color=mesh_color,
                camera_center=[K[0, 2], K[1, 2]],
            )
        else:
            # Weak-perspective HMR2: pred_cam_t lives in 256-crop space. Convert
            # to full-image translation via the same CLIFF reparameterization the
            # perspective branch uses, so the mesh lands at the bbox location.
            pc = torch.from_numpy(np.asarray(result["pred_cam"])).float().unsqueeze(0)
            bbx_t = torch.from_numpy(np.asarray(bbx_xys)).float().unsqueeze(0)
            K_t = torch.from_numpy(np.asarray(K_fullimg)).float().unsqueeze(0)
            cam_t_full = compute_transl_full_cam(pc, bbx_t, K_t)[0].numpy()
            focal = float(K_fullimg[0, 0])
            cx = float(K_fullimg[0, 2])
            cy = float(K_fullimg[1, 2])
            if renderer is None:
                renderer = _get_renderer_class()(focal_length=focal, faces=faces)
            else:
                renderer.focal_length = focal
            return renderer(
                vertices,
                cam_t_full,
                full_img_rgb,
                mesh_base_color=mesh_color,
                camera_center=[cx, cy],
            )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _open_video_writer(path, width, height, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))


# ---------------------------------------------------------------------------
# Clip rendering
# ---------------------------------------------------------------------------


def render_clip(
    out_dir,
    frames,
    frame_names,
    results_per_hand,
    crops_per_hand,
    bbx_per_hand,
    faces_by_hand,
    hands,
    W,
    H,
    camera_fullimg,
    fps,
    mesh_color,
    save_images,
    save_videos,
    draw_skeleton=False,
    no_mesh=False,
    display_size=None,
    save_crop_overlays=False,
):
    """Render bbox/crop/full-image overlays and videos for a hand-inference clip.

    Takes already-computed results as input — no bbox detection or model
    inference happens here. NPZ saving is the caller's responsibility — call
    save_hand_results directly.

    Right-hand handling: results for the right hand are expected to already
    be mirrored back into the original camera frame with right-hand SOMA
    topology (see inference.preprocessing.mirroring.prepare_right_hand_model_input),
    so this renderer uses right-hand faces directly (no reversed face winding).

    ``frames``, ``W``, ``H`` and ``camera_fullimg`` all describe the raster
    inference ran on. ``display_size`` optionally gives a different
    ``(width, height)`` to write the full-image outputs at, which is how a
    run that rectified the pixel aspect ratio puts its overlays back on the
    source video's resolution; ``None`` writes at ``W x H``.
    ``camera_fullimg`` accepts a pinhole matrix or a camera object; mesh
    overlays are supported for PINHOLE cameras.

    ``save_crop_overlays`` gates just the per-hand ``crop_overlay_<hand>``
    outputs (images and/or video, per ``save_images``/``save_videos``); off
    by default. ``bbox_viz`` and ``fullimg_overlay`` are unaffected either
    way. Crop overlays are mainly a debugging aid: the crop window is
    centered on that frame's raw detected bbox, which -- unlike the hand
    pose/mesh itself -- is never touched by upstream temporal smoothing, so
    the crop can visibly jitter frame to frame even when the mesh inside it
    is smooth.
    """
    out_dir = Path(out_dir)
    mesh_color = tuple(mesh_color)
    if isinstance(camera_fullimg, np.ndarray):
        K_fullimg = camera_fullimg
    else:
        no_mesh = no_mesh or camera_fullimg.model_name != "PINHOLE"
        K_fullimg = camera_fullimg.K
    Renderer = _get_renderer_class()
    # Overlays are composited at W x H -- the raster inference ran on -- and
    # resized once at write time. Rendering into the output size instead
    # would mean projecting through a K that does not describe these frames.
    out_W, out_H = display_size if display_size is not None else (W, H)

    def _to_display(img):
        if img.shape[1] == out_W and img.shape[0] == out_H:
            return img
        return cv2.resize(img, (out_W, out_H), interpolation=cv2.INTER_AREA)

    # --- Bbox visualization (both hands on same image) ---
    colors = {"left": (0, 255, 0), "right": (255, 0, 0)}  # green / blue (BGR)
    bbox_viz_dir = out_dir / "bbox_viz"
    bbox_writer = None
    if save_images:
        bbox_viz_dir.mkdir(parents=True, exist_ok=True)
    if save_videos and len(frames) > 1:
        bbox_writer = _open_video_writer(out_dir / "bbox_viz.mp4", out_W, out_H, fps)
    for fi, frame in enumerate(frames):
        vis = frame.copy()
        for h in hands:
            if bbx_per_hand[h][fi] is not None:
                draw_bbox_on_image(vis, bbx_per_hand[h][fi], color=colors[h], label=h)
        vis = _to_display(vis)
        if save_images:
            cv2.imwrite(str(bbox_viz_dir / f"{frame_names[fi]}_bbox.jpg"), vis)
        if bbox_writer is not None:
            bbox_writer.write(vis)
    if bbox_writer is not None:
        bbox_writer.release()
    msg_parts = []
    if save_images:
        msg_parts.append(f"{bbox_viz_dir}/")
    if bbox_writer is not None:
        msg_parts.append(f"{out_dir / 'bbox_viz.mp4'}")
    print(f"[render_clip] Bbox viz: {' + '.join(msg_parts)}")

    # --- Crop overlays (per hand, separate videos) ---
    save_crop_images = save_images and save_crop_overlays
    save_crop_videos = save_videos and save_crop_overlays
    if not save_crop_overlays:
        print(
            "[render_clip] Crop overlays: skipped (pass --save_crop_overlays to enable)"
        )
    for h in hands if save_crop_overlays else ():
        results = results_per_hand[h]
        crops, bbxs_viz, Ks_viz = crops_per_hand[h]
        crop_out_dir = out_dir / f"crop_overlay_{h}"
        if save_crop_images:
            crop_out_dir.mkdir(parents=True, exist_ok=True)
        crop_writer = None
        if save_crop_videos and len(results) > 1:
            crop_writer = _open_video_writer(
                out_dir / f"crop_overlay_{h}.mp4",
                IMG_SIZE,
                IMG_SIZE,
                fps,
            )
        renderer = Renderer(focal_length=1.0, faces=faces_by_hand[h])
        for fi, (result, img_crop, bbx, K) in enumerate(
            zip(results, crops, bbxs_viz, Ks_viz)
        ):
            if result is None or img_crop is None:
                continue
            if not (save_crop_images or crop_writer is not None):
                continue
            overlay = render_crop_overlay(
                result,
                img_crop,
                bbx,
                K,
                faces_by_hand[h],
                "egohand",
                mesh_color=mesh_color,
                renderer=renderer,
            )
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            if save_crop_images:
                cv2.imwrite(
                    str(crop_out_dir / f"{frame_names[fi]}_crop.jpg"), overlay_bgr
                )
            if crop_writer is not None:
                crop_writer.write(overlay_bgr)
        if crop_writer is not None:
            crop_writer.release()
        renderer.close()
        msg_parts = []
        if save_crop_images:
            msg_parts.append(f"{crop_out_dir}/")
        if crop_writer is not None:
            msg_parts.append(f"{out_dir / f'crop_overlay_{h}.mp4'}")
        print(f"[render_clip] Crop overlays ({h}): {' + '.join(msg_parts)}")

    # --- Full-image overlays (both hands chained onto same frame) ---
    full_renderers = (
        {}
        if no_mesh
        else {h: Renderer(focal_length=1.0, faces=faces_by_hand[h]) for h in hands}
    )
    full_out_dir = out_dir / "fullimg_overlay"
    full_writer = None
    if save_images:
        full_out_dir.mkdir(parents=True, exist_ok=True)
    if save_videos and len(frames) > 1:
        full_writer = _open_video_writer(
            out_dir / "fullimg_overlay.mp4", out_W, out_H, fps
        )
    skel_colors = {"left": (0, 255, 0), "right": (0, 0, 255)}
    for fi, frame in enumerate(frames):
        full_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for h in hands:
            result = results_per_hand[h][fi]
            if result is None:
                continue
            if not no_mesh:
                full_img_rgb = draw_hand_mesh(
                    result,
                    full_img_rgb,
                    K_fullimg,
                    None,
                    faces_by_hand[h],
                    "egohand",
                    mesh_color=mesh_color,
                    renderer=full_renderers[h],
                )
            if draw_skeleton:
                draw_hand_skeleton(
                    full_img_rgb,
                    result["pred_keypoints_2d"],
                    skel_colors[h],
                    bbx_per_hand[h][fi],
                )
        overlay_bgr = _to_display(cv2.cvtColor(full_img_rgb, cv2.COLOR_RGB2BGR))
        if save_images:
            cv2.imwrite(
                str(full_out_dir / f"{frame_names[fi]}_fullimg.jpg"), overlay_bgr
            )
        if full_writer is not None:
            full_writer.write(overlay_bgr)
    if full_writer is not None:
        full_writer.release()
    for r in full_renderers.values():
        r.close()
    msg_parts = []
    if save_images:
        msg_parts.append(f"{full_out_dir}/")
    if full_writer is not None:
        msg_parts.append(f"{out_dir / 'fullimg_overlay.mp4'}")
    print(f"[render_clip] Full-image overlays: {' + '.join(msg_parts)}")

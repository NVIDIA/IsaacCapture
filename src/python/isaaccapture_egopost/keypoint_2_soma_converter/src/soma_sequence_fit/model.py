# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Native SOMA adapter and bind-relative articulation prior."""

from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from .data import sha256_file
from .geometry import DEFAULT_JOINT_MAP, rotvec_matrix


ASSET_FILES = (
    "SOMAHand.npz",
    "SOMA_neutral.npz",
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
)

MHR_ASSET_FILES = (
    "MHR/mhr_model_lod1.pt",
    "MHR/base_body_lod1.obj",
    "MHR/SOMA_wrap_lod1.obj",
)


def verify_assets(directory, files):
    """Reject assets whose SHA-256 differs from assets.json (SOMA-X v0.3.0)."""
    expected = json.loads(Path(__file__).with_name("assets.json").read_text())
    for name in files:
        if name not in expected:
            raise ValueError(f"No expected SHA-256 for native asset {name!r}.")
        actual = sha256_file(Path(directory) / name)
        if actual != expected[name]:
            raise ValueError(
                f"SHA-256 mismatch for native asset {name!r}: expected "
                f"{expected[name]}, got {actual}. Use SOMA-X v0.3.0 assets."
            )


def fetch_assets(destination, revision="v0.3.0", identity_backend="soma"):
    """Download the requested native identity assets; never download MANO assets."""
    from huggingface_hub import hf_hub_download

    if identity_backend not in {"soma", "mhr"}:
        raise ValueError("identity_backend must be 'soma' or 'mhr'.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = ASSET_FILES + (MHR_ASSET_FILES if identity_backend == "mhr" else ())
    for name in files:
        hf_hub_download("nvidia/SOMA-X", name, revision=revision, local_dir=destination)
    verify_assets(destination, files)
    return destination


class NativeHand(nn.Module):
    def __init__(
        self,
        assets,
        side,
        device="cpu",
        lod="low",
        components=32,
        mapping=None,
        identity_backend="soma",
        wrist_source="landmark",
    ):
        super().__init__()
        assets = Path(assets)
        if identity_backend not in {"soma", "mhr"}:
            raise ValueError("identity_backend must be 'soma' or 'mhr'.")
        self.asset_files = ASSET_FILES + (
            MHR_ASSET_FILES if identity_backend == "mhr" else ()
        )
        missing = [n for n in self.asset_files if not (assets / n).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing native assets: {missing}. Run the assets command first."
            )
        verify_assets(assets, self.asset_files)
        try:
            from soma import SOMAHandLayer
        except ImportError as e:
            raise ImportError(
                "Install dependencies with python -m pip install -r requirements.txt."
            ) from e
        self.layer = SOMAHandLayer(
            data_root=assets,
            hand_type=side,
            device=device,
            identity_model_type=identity_backend,
            mode="torch",
            lod=lod,
            correctives_model_path=None,
        )
        self.side, self.lod = side, lod
        self.identity_backend = identity_backend
        self.nshape = self.layer.num_shape_components
        identity_scales = self.layer.identity_model.num_scale_params
        self.nscale = int(
            identity_scales
            if identity_scales is not None
            else self.layer.NUM_BONE_SCALE_PARAMS
        )
        self.joint_names = list(self.layer.rig_data["joint_names"])
        self.register_buffer("faces", self.layer.faces.long())
        with np.load(assets / "SOMAHand.npz", allow_pickle=False) as d:
            prefix = side + "_pose_"
            metadata = json.loads(str(d["pose_pca_metadata"].item()))
            if metadata.get("rotation_repr") != "expmap" or not metadata.get(
                "identity_at_bind_pose"
            ):
                raise ValueError("Unsupported SOMA pose prior convention.")
            if (
                list(d[prefix + "reference_joint_names"].astype(str))
                != self.joint_names
            ):
                raise ValueError("Pose prior/runtime joint ordering mismatch.")
            if not np.array_equal(
                d[prefix + "reference_parent_ids"],
                self.layer.joint_parent_ids.cpu().numpy(),
            ):
                raise ValueError("Pose prior/runtime hierarchy mismatch.")
            if not np.allclose(
                d[prefix + "reference_orient"],
                self.layer.bind_pose_world.cpu().numpy()[:, :3, :3],
                atol=1e-5,
            ):
                raise ValueError("Pose prior/runtime bind frames mismatch.")
            if not 1 <= components <= 72:
                raise ValueError("Pose components must be in [1,72].")
            native_basis = np.array(d[prefix + "components"])
            native_count = len(native_basis)
            n = min(components, native_count)
            pose_basis = native_basis[:n] * np.sqrt(d[prefix + "eigenvalues"][:n, None])
            if components > native_count:
                # Complete the native prior with orthogonal articulation directions.
                # This spans joint rotations that the truncated PCA cannot express;
                # no other model or reference supplies these directions.
                _, _, vh = np.linalg.svd(
                    native_basis.astype(np.float64), full_matrices=True
                )
                residual = vh[native_count:components].astype(np.float32) * 0.25
                pose_basis = np.concatenate([pose_basis, residual], axis=0)
            self.npose = len(pose_basis)
            for name, value in {
                "mean": d[prefix + "mean"].reshape(72),
                "basis": pose_basis,
                "orient": d[prefix + "reference_orient"],
                "parent_orient": d[prefix + "reference_orient_parent"],
            }.items():
                self.register_buffer(
                    name,
                    torch.as_tensor(
                        np.array(value), dtype=torch.float32, device=device
                    ),
                )
        spec = json.loads(Path(mapping).read_text()) if mapping else {}
        if "_calibration" in spec:
            raise ValueError("Reference calibration is not accepted by this fitter.")
        spec = spec.get(side, spec)
        if wrist_source not in {"palm", "landmark"}:
            raise ValueError("wrist_source must be palm or landmark")
        self.wrist_source = wrist_source
        joint_map = spec.get("joint_indices", DEFAULT_JOINT_MAP)
        offsets = spec.get("local_offsets_m", np.zeros((21, 3)).tolist())
        # Source wrist landmarks can use a different anatomical convention.
        # Infer the native root from palm MCPs and the native rig; never shift
        # the observation centers using reference-derived calibration offsets.
        weights = np.ones(21, np.float32)
        weights[0] = 0.0 if wrist_source == "palm" else 4.0
        if wrist_source == "palm":
            weights[[1, 5, 9, 13, 17]] = 4.0
        weights = spec.get("weights", weights.tolist())
        if np.any(np.asarray(offsets) != 0):
            raise ValueError(
                "Reference-free fitting requires zero local observation offsets."
            )
        if len(joint_map) != 21 or any(
            int(i) != i or i < 0 or i >= 25 for i in joint_map
        ):
            raise ValueError("joint_indices must contain 21 integers in [0,24].")
        if np.shape(offsets) != (21, 3) or not np.isfinite(offsets).all():
            raise ValueError("local_offsets_m must be finite (21,3).")
        if (
            np.shape(weights) != (21,)
            or not np.isfinite(weights).all()
            or np.any(np.array(weights) < 0)
        ):
            raise ValueError("weights must be nonnegative finite (21,).")
        self.register_buffer("joint_map", torch.tensor(joint_map, device=device))
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "observation_weights",
            torch.tensor(weights, dtype=torch.float32, device=device),
        )
        for p in self.layer.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def native_palm_ratios(self):
        """Native neutral palm radii divided by MCP width; no input wrist used."""
        device = self.basis.device
        self.prepare_identity(
            torch.zeros(1, self.nshape, device=device), torch.zeros(1, device=device)
        )
        output = self.layer.pose(
            torch.zeros(1, 25, 3, device=device),
            pose2rot=True,
            absolute_pose=False,
            apply_correctives=False,
            global_translation=torch.zeros(1, 3, device=device),
        )
        q = output["joints"][:, self.joint_map]
        width = torch.linalg.vector_norm(q[:, 5] - q[:, 17], dim=-1)
        radii = torch.linalg.vector_norm(
            q[:, [5, 9, 13, 17]] - output["joints"][:, :1], dim=-1
        )
        return (radii / width[:, None].clamp_min(1e-6))[0]

    def decode(self, latent, root_rotvec):
        relative = rotvec_matrix((self.mean + latent @ self.basis).reshape(-1, 24, 3))
        eye = torch.eye(3, device=relative.device, dtype=relative.dtype).expand(
            len(latent), 1, 3, 3
        )
        relative = torch.cat((eye, relative), dim=1)
        absolute = self.parent_orient.transpose(-1, -2) @ relative @ self.orient
        root = rotvec_matrix(root_rotvec) @ absolute[:, 0]
        return torch.cat((root[:, None], absolute[:, 1:]), dim=1)

    def _scale_params(self, bone_log_scales):
        if bone_log_scales is None:
            return None
        # SOMA uses multiplicative bone scales; MHR exposes its native additive
        # scale vector, whose neutral value is zero.
        return (
            bone_log_scales + 0.0
            if self.identity_backend == "mhr"
            else bone_log_scales.exp()
        )

    def prepare_identity(self, beta, log_scale, bone_log_scales=None):
        """Prepare one fixed identity for any number of pose-only evaluations."""
        self.layer.prepare_identity(
            beta,
            scale_params=self._scale_params(bone_log_scales),
            repose_to_bind_pose=False,
            global_scale=log_scale.exp().reshape(-1),
        )

    def pose_prepared(self, latent, root_rotvec, translation):
        """Pose the currently prepared identity without rerunning its backend."""
        rotations = self.decode(latent, root_rotvec)
        output = self.layer.pose(
            rotations,
            pose2rot=False,
            absolute_pose=True,
            global_translation=translation,
            apply_correctives=False,
        )
        transforms = output["transforms"][:, self.joint_map]
        observations = output["joints"][:, self.joint_map] + (
            transforms[:, :, :3, :3] @ self.offsets[None, :, :, None]
        ).squeeze(-1)
        return output["vertices"], output["joints"], observations, rotations

    def forward(
        self, beta, latent, root_rotvec, translation, log_scale, bone_log_scales=None
    ):
        # A trainable identity graph is prepared for this forward only. Pose-only
        # callers use prepare_identity() + pose_prepared() to reuse a detached cache.
        self.prepare_identity(beta, log_scale, bone_log_scales)
        return self.pose_prepared(latent, root_rotvec, translation)

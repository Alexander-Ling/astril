"""Geometry and transform helpers shared by the GPU registration backend (ITK v4 conventions).

Ported from ITK 5.4 (the version bundled with SimpleITK 2.5.5):
  - MattesMutualInformationImageToImageMetricv4 (+ GetValueAndDerivativeThreader), global-support transforms
  - CorrelationImageToImageMetricv4 (+ HelperThreader, GetValueAndDerivativeThreader)
  - ImageToImageMetricv4 point handling: fixed/moving masks (ImageMaskSpatialObject: nearest voxel != 0),
    interpolator IsInsideBuffer (continuous index in [-0.5, size-0.5)), linear interpolation, moving-image
    gradient from GradientRecursiveGaussian(sigma = max moving spacing, normalize across scale, image direction)
    sampled with linear interpolation.

Conventions follow ITK exactly:
  - The virtual domain is the fixed image grid (fixed transform = identity), dense sampling over every voxel.
  - Points are ITK physical points (LPS); image arrays are z,y,x as returned by sitk.GetArrayFromImage.
  - The returned "derivative" is ITK's v4 derivative, i.e. already negated for the optimizers
    (optimizers do params += learning_rate * derivative).

Pyramid smoothing/shrinking and the moving gradient image are computed with SimpleITK itself (ITK filters), so the
inputs the GPU sees are the same images ITK's metric sees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
import torch

DEFAULT_CHUNK = 2_000_000  # points per chunk (bounds memory for the per-point derivative terms)
HIST_COPIES = 64           # private joint-histogram copies for the fast path (atomic-contention relief)


# ------------------------------------------------------------------------------------------------ geometry
@dataclass
class GpuImage:
    """A scalar or vector SimpleITK image on the GPU plus its index<->physical geometry."""
    data: torch.Tensor          # [C, Z, Y, X] float32
    size_xyz: torch.Tensor      # (3,) float64, sizes in x,y,z order
    index_to_phys: torch.Tensor  # (3,3) float64: phys = origin + M @ idx_xyz
    origin: torch.Tensor        # (3,) float64
    phys_to_index: torch.Tensor  # (3,3) float64 inverse of index_to_phys

    @classmethod
    def from_sitk(cls, img: sitk.Image, device) -> "GpuImage":
        arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
        if img.GetNumberOfComponentsPerPixel() == 1:
            arr = arr[None]
        else:
            arr = np.moveaxis(arr, -1, 0)
        sp = np.asarray(img.GetSpacing(), dtype=np.float64)
        d = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
        m = d @ np.diag(sp)
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
        return cls(data=t(arr), size_xyz=t(np.asarray(img.GetSize(), dtype=np.float64)), index_to_phys=t(m),
                   origin=t(np.asarray(img.GetOrigin(), dtype=np.float64)), phys_to_index=t(np.linalg.inv(m)))

    def cont_index(self, pts: torch.Tensor) -> torch.Tensor:
        """Physical points [N,3] (float64) -> continuous index [N,3] (x,y,z)."""
        return (pts - self.origin) @ self.phys_to_index.T

    def inside_buffer(self, cidx: torch.Tensor) -> torch.Tensor:
        """ITK ImageFunction::IsInsideBuffer for continuous indices: start-0.5 <= i < end-0.5 (NaN -> False)."""
        return ((cidx >= -0.5) & (cidx < self.size_xyz - 0.5)).all(dim=1)

    def sample_linear(self, cidx: torch.Tensor) -> torch.Tensor:
        """Linear interpolation at continuous indices [N,3] -> [N,C] (edge-clamped, like ITK's linear interpolator)."""
        n = cidx.shape[0]
        norm = (cidx / (self.size_xyz - 1.0)) * 2.0 - 1.0  # align_corners=True mapping, x,y,z order
        grid = norm.to(torch.float32).view(1, 1, 1, n, 3)
        out = torch.nn.functional.grid_sample(self.data[None], grid, mode="bilinear", padding_mode="border",
                                              align_corners=True)
        return out.view(self.data.shape[0], n).T

    def mask_inside(self, pts: torch.Tensor) -> torch.Tensor:
        """ImageMaskSpatialObject::IsInsideInWorldSpace: nearest voxel (round half up) inside region and != 0."""
        idx = torch.floor(self.cont_index(pts) + 0.5)
        ok = ((idx >= 0) & (idx < self.size_xyz)).all(dim=1)
        out = torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
        if ok.any():
            i = idx[ok].long()
            out[ok] = self.data[0, i[:, 2], i[:, 1], i[:, 0]] != 0
        return out

    def voxel_points(self, flat_idx: torch.Tensor) -> torch.Tensor:
        """Physical points of voxels given flat (z,y,x C-order) indices."""
        sx, sy = int(self.size_xyz[0]), int(self.size_xyz[1])
        x = flat_idx % sx
        y = (flat_idx // sx) % sy
        z = flat_idx // (sx * sy)
        ijk = torch.stack([x, y, z], dim=1).to(torch.float64)
        return self.origin + ijk @ self.index_to_phys.T


# ---------------------------------------------------------------------------------------------- transforms
class TorchTransform:
    """ITK-compatible global transform: T(x) = M(p) (x - c) + c + t(p). Jacobian dT/dp computed from (M, t)."""

    def __init__(self, kind: str, params, center, device, compute_zyx: bool = False):
        self.kind = kind
        self.device = device
        self.compute_zyx = bool(compute_zyx)
        self.params = torch.as_tensor(np.asarray(params, dtype=np.float64), device=device)
        self.center = torch.as_tensor(np.asarray(center, dtype=np.float64), device=device)

    @classmethod
    def from_sitk(cls, tfm: sitk.Transform, device) -> "TorchTransform":
        tfm = tfm.Downcast()
        name = tfm.GetName()
        if name == "CompositeTransform":
            if tfm.GetNumberOfTransforms() != 1:
                raise NotImplementedError("composite transforms with >1 member are not supported yet")
            tfm = tfm.GetNthTransform(0).Downcast()
            name = tfm.GetName()
        p, fp = tfm.GetParameters(), tfm.GetFixedParameters()
        if name == "Euler3DTransform":
            return cls("euler3d", p, fp[:3], device, compute_zyx=bool(fp[3]) if len(fp) > 3 else False)
        if name == "AffineTransform":
            return cls("affine", p, fp[:3], device)
        if name == "TranslationTransform":
            return cls("translation", p, (0.0, 0.0, 0.0), device)
        raise NotImplementedError(f"transform {name} not supported")

    def matrix_offset(self, p: torch.Tensor):
        if self.kind == "euler3d":
            ax, ay, az = p[0], p[1], p[2]
            one, zero = torch.ones((), dtype=p.dtype, device=p.device), torch.zeros((), dtype=p.dtype, device=p.device)
            cx, sx, cy, sy, cz, sz = torch.cos(ax), torch.sin(ax), torch.cos(ay), torch.sin(ay), torch.cos(az), torch.sin(az)
            rx = torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, cx, -sx]), torch.stack([zero, sx, cx])])
            ry = torch.stack([torch.stack([cy, zero, sy]), torch.stack([zero, one, zero]), torch.stack([-sy, zero, cy])])
            rz = torch.stack([torch.stack([cz, -sz, zero]), torch.stack([sz, cz, zero]), torch.stack([zero, zero, one])])
            m = rz @ ry @ rx if self.compute_zyx else rz @ rx @ ry
            return m, p[3:6]
        if self.kind == "affine":
            return p[:9].reshape(3, 3), p[9:12]
        if self.kind == "translation":
            return torch.eye(3, dtype=p.dtype, device=p.device), p[:3]
        raise NotImplementedError(self.kind)

    def transform_and_jacobian(self, pts: torch.Tensor, need_jacobian: bool):
        m, t = self.matrix_offset(self.params)
        rel = pts - self.center
        mapped = rel @ m.T + self.center + t
        if not need_jacobian:
            return mapped, None
        dm, dt = torch.autograd.functional.jacobian(self.matrix_offset, self.params)  # [3,3,P], [3,P]
        jac = torch.einsum("ijp,nj->nip", dm, rel) + dt[None]  # [N,3,P]
        return mapped, jac

    @property
    def center_np(self) -> np.ndarray:
        return self.center.cpu().numpy() if not hasattr(self, "_center_np") else self._center_np

    def host_matrix_offset_jacobian(self):
        """Host (numpy, float64) M, off (mapped = M x + off), dM/dp [3,3,P], d(off)/dp [3,P] for x -> M(x-c)+c+t."""
        if not hasattr(self, "_center_np"):
            self._center_np = self.center.cpu().numpy()
        p = self.params.cpu().numpy() if not hasattr(self, "_params_np") else self._params_np
        c = self._center_np
        P = p.size
        if self.kind == "euler3d":
            ax, ay, az = p[:3]
            cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
            rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            drx = np.array([[0, 0, 0], [0, -sx, -cx], [0, cx, -sx]])
            dry = np.array([[-sy, 0, cy], [0, 0, 0], [-cy, 0, -sy]])
            drz = np.array([[-sz, -cz, 0], [cz, -sz, 0], [0, 0, 0]])
            if self.compute_zyx:
                m = rz @ ry @ rx
                dms = [rz @ ry @ drx, rz @ dry @ rx, drz @ ry @ rx]
            else:
                m = rz @ rx @ ry
                dms = [rz @ drx @ ry, rz @ rx @ dry, drz @ rx @ ry]
            t = p[3:6]
            dm = np.zeros((3, 3, P)); dt = np.zeros((3, P))
            for k in range(3):
                dm[:, :, k] = dms[k]
            dt[:, 3:6] = np.eye(3)
        elif self.kind == "affine":
            m, t = p[:9].reshape(3, 3), p[9:12]
            dm = np.zeros((3, 3, P)); dt = np.zeros((3, P))
            for k in range(9):
                dm[k // 3, k % 3, k] = 1.0
            dt[:, 9:12] = np.eye(3)
        elif self.kind == "translation":
            m, t = np.eye(3), p[:3]
            dm = np.zeros((3, 3, P)); dt = np.eye(3)
        else:
            raise NotImplementedError(self.kind)
        off = c + t - m @ c
        # d(off)/dp = dt - dM c ; the metric derivative uses J = dM (x - c) + dt, handled in _contract_host.
        return m, off, dm, dt

    def set_params(self, params):
        self._params_np = np.asarray(params, dtype=np.float64).copy()
        self.params = torch.as_tensor(self._params_np, device=self.device)

    @property
    def n_params(self) -> int:
        return int(self.params.numel())

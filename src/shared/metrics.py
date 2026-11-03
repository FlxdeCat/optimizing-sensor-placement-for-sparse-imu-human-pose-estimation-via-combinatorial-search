"""Thesis Table I metrics. Positions enter in metres; rotations are local.

Output units: degrees, centimetres, km/s^3, seconds. All 24 joints, including
the root, are counted. Root alignment removes translation, never heading.
Search uses the frame-weighted local error; report tables retain sequence
metrics for descriptive sequence summaries. A root has no parent: its local rotation
is measured relative to the world. See README.md section 7 for reporting conventions and the functions below for metric definitions.
"""
from __future__ import annotations

import math
import logging
import time
from pathlib import Path
from typing import Any

import torch

from config import Deadline, PARENTS, SIP_ERROR_JOINTS
from .vendor_adapter import vendor
from .storage import atomic_write


def _matrices(value: Any) -> torch.Tensor:
    result = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if result.shape[-2:] != (3, 3) or not torch.isfinite(result).all():
        raise ValueError("Rotations must be finite matrices [...,3,3].")
    return result


def project_so3(value: Any, *, chunk_size: int = 4096) -> torch.Tensor:
    """Nearest proper rotation U diag(1,1,det(U V^T)) V^T, on CPU float64.

Projection belongs only to inference/metrics, never to the matrix-MSE loss.
Rank-deficient inputs have nonunique closest rotations; diagnostics count them.
"""
    matrix = _matrices(value)
    if chunk_size < 1:
        raise ValueError("Projection chunk size must be positive.")
    flat = matrix.reshape(-1, 3, 3)
    output = []
    for chunk in flat.split(chunk_size):
        u, _, vh = torch.linalg.svd(chunk)
        signs = torch.ones((len(chunk), 3), dtype=chunk.dtype)
        signs[:, -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0)
        output.append((u * signs[:, None, :]) @ vh)
    if not output:
        raise ValueError("At least one rotation matrix is required.")
    return torch.cat(output).reshape(matrix.shape)


def check_so3(value: Any, tolerance: float = 1e-3) -> torch.Tensor:
    matrix = _matrices(value)
    identity = torch.eye(3, dtype=matrix.dtype)
    if not torch.allclose(matrix.transpose(-1, -2) @ matrix, identity.expand_as(matrix), atol=tolerance, rtol=0):
        raise ValueError("Ground truth or projected rotations are not orthonormal.")
    if bool((torch.linalg.det(matrix) - 1).abs().gt(tolerance).any()):
        raise ValueError("Rotations must have determinant +1.")
    return matrix


def geodesic_degrees(prediction: Any, target: Any) -> torch.Tensor:
    """SO(3) angle atan2(sin(theta), cos(theta)), equivalent to clipped acos.

Using the antisymmetric part improves accuracy at zero degrees. At pi the
nonnegative sine and negative cosine correctly give 180 degrees.
"""
    pred, truth = check_so3(prediction), check_so3(target)
    if pred.shape != truth.shape:
        raise ValueError("Rotation arrays must have matching shapes.")
    relative = pred.transpose(-1, -2) @ truth
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
    sine = torch.linalg.vector_norm(skew, dim=-1) / 2
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.atan2(sine, cosine.clamp(-1, 1)) * (180 / math.pi)


def global_rotations(local: Any) -> torch.Tensor:
    local = check_so3(local)
    if local.ndim < 3 or local.shape[-3] != 24:
        raise ValueError("SMPL requires 24 local rotations.")
    values = [local[..., 0, :, :]]
    for joint in range(1, 24):
        values.append(values[PARENTS[joint]] @ local[..., joint, :, :])
    return torch.stack(values, dim=-3)


def rotation_errors(prediction: Any, target: Any, *, project: bool = True) -> dict[str, float]:
    pred = project_so3(prediction) if project else check_so3(prediction)
    truth = check_so3(target)
    local = geodesic_degrees(pred, truth)
    global_error = geodesic_degrees(global_rotations(pred), global_rotations(truth))
    return {"local_rotation_deg": float(local.mean()),
            "global_rotation_deg": float(global_error.mean()),
            "sip_deg": float(global_error[..., list(SIP_ERROR_JOINTS)].mean())}


def _positions(value: Any) -> torch.Tensor:
    result = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if result.ndim != 3 or result.shape[-1] != 3 or result.shape[0] < 1 or not torch.isfinite(result).all():
        raise ValueError("Positions must be finite [T,N,3] in metres.")
    return result


def mpjpe_cm(prediction: Any, target: Any) -> float:
    pred, truth = _positions(prediction), _positions(target)
    if pred.shape != truth.shape or pred.shape[1] != 24:
        raise ValueError("MPJPE needs matching [T,24,3] joint positions.")
    difference = (pred - pred[:, :1]) - (truth - truth[:, :1])
    return float(torch.linalg.vector_norm(difference, dim=-1).mean() * 100)


def mesh_error_cm(prediction: Any, target: Any, predicted_root: Any, target_root: Any) -> float:
    pred, truth = _positions(prediction), _positions(target)
    p_root = torch.as_tensor(predicted_root).detach().to(device="cpu", dtype=torch.float64)
    t_root = torch.as_tensor(target_root).detach().to(device="cpu", dtype=torch.float64)
    if pred.shape != truth.shape or p_root.shape != (len(pred), 3) or t_root.shape != p_root.shape:
        raise ValueError("Mesh error requires matched vertices and explicit [T,3] pelvis roots.")
    if not torch.isfinite(p_root).all() or not torch.isfinite(t_root).all():
        raise ValueError("Mesh roots must be finite.")
    difference = (pred - p_root[:, None]) - (truth - t_root[:, None])
    return float(torch.linalg.vector_norm(difference, dim=-1).mean() * 100)


def jitter_km_s3(prediction: Any, fps: float = 60.0) -> float | None:
    joints = _positions(prediction)
    if joints.shape[1] != 24 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Jitter requires [T,24,3] joints and a positive finite frame rate.")
    if len(joints) < 4:
        return None
    aligned = joints - joints[:, :1]
    jerk = aligned[3:] - 3 * aligned[2:-1] + 3 * aligned[1:-2] - aligned[:-3]
    return float(torch.linalg.vector_norm(jerk, dim=-1).mean() * fps ** 3 / 1000)


def training_time_seconds(seconds: float) -> float:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Training time must be finite nonnegative active wall-clock seconds.")
    return float(seconds)


class PoseMetrics:
    """Mean-shape, male SMPL LBS, matching the upstream default (no pose blend).

FK stays on CPU in small chunks so scoring cannot compete with GPU training
memory. The upstream API is forward_kinematics(pose, shape=None, tran=None,
calc_mesh=False); None shape denotes ten zero coefficients (mean shape).
"""
    def __init__(self, smpl_path: Path, *, chunk_frames: int = 32, fps: float = 60.0):
        if chunk_frames < 1:
            raise ValueError("Metric FK chunk size must be positive.")
        self.body = vendor("cpu").model(smpl_path, use_pose_blendshape=False)
        self.chunk_frames, self.fps = chunk_frames, fps

    @torch.no_grad()
    def __call__(self, prediction: Any, target: Any, *, deadline: Deadline | None = None,
                 progress_path: Path | None = None, identity: str = "", progress_interval_seconds: float = 60.0) -> dict[str, Any]:
        if not math.isfinite(progress_interval_seconds) or progress_interval_seconds<=0:raise ValueError("Metric save interval must be positive.")
        raw = _matrices(prediction)
        pred, truth = project_so3(raw), check_so3(target)
        if pred.shape != truth.shape or pred.ndim != 4 or pred.shape[1] != 24:
            raise ValueError("Pose metrics need matching nonempty [T,24,3,3] arrays.")
        result: dict[str, Any] = rotation_errors(pred, truth, project=False)
        position_sum = mesh_sum = 0.0
        predicted_joints = []
        vertices = 0
        next_frame = 0
        if progress_path is not None and progress_path.exists():
            state = torch.load(progress_path, map_location="cpu", weights_only=True)
            if state["identity"] != identity or state["frames"] != len(pred):
                raise RuntimeError("Metric progress identity changed; use a new run root.")
            next_frame = state["next_frame"]
            position_sum, mesh_sum = state["position_sum"], state["mesh_sum"]
            predicted_joints, vertices = state["predicted_joints"], state["vertices"]
        last_save = time.monotonic()
        for start in range(next_frame, len(pred), self.chunk_frames):
            if deadline is not None:
                deadline.check()
            p = pred[start:start + self.chunk_frames].float().contiguous()
            t = truth[start:start + self.chunk_frames].float().contiguous()
            _, p_joint, p_mesh = self.body.forward_kinematics(p, calc_mesh=True)
            _, t_joint, t_mesh = self.body.forward_kinematics(t, calc_mesh=True)
            count = len(p)
            position_sum += mpjpe_cm(p_joint, t_joint) * count
            mesh_sum += mesh_error_cm(p_mesh, t_mesh, p_joint[:, 0], t_joint[:, 0]) * count
            predicted_joints.append(p_joint)
            vertices = p_mesh.shape[1]
            if progress_path is not None and time.monotonic() - last_save >= progress_interval_seconds:
                last_save = time.monotonic()
                state = {"identity": identity, "frames": len(pred), "next_frame": start + count,
                         "position_sum": position_sum, "mesh_sum": mesh_sum,
                         "predicted_joints": predicted_joints, "vertices": vertices}
                atomic_write(progress_path, lambda stream: torch.save(state, stream))
                logging.getLogger("imuplace").info("METRICS FK progress %d/%d frames",start+count,len(pred))
        singular = torch.linalg.svdvals(raw)
        result.update(mpjpe_cm=position_sum / len(pred), mesh_cm=mesh_sum / len(pred),
                      jitter_km_s3=jitter_km_s3(torch.cat(predicted_joints), self.fps),
                      frames=len(pred), mesh_vertices=vertices,
                      rank_deficient_fraction=float((singular[..., -1] < 1e-8).double().mean()))
        if progress_path is not None:
            progress_path.unlink(missing_ok=True)
        return result


def aggregate_sequences(rows: list[dict[str, Any]], *, frame_weighted: bool = False) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty sequence cohort.")
    names = ("local_rotation_deg", "global_rotation_deg", "sip_deg", "mpjpe_cm", "mesh_cm", "jitter_km_s3")
    output: dict[str, Any] = {"sequences": len(rows), "frames": sum(row["frames"] for row in rows)}
    for name in names:
        available = [row for row in rows if row.get(name) is not None]
        weights = [row["frames"] if frame_weighted else 1 for row in available]
        output[name] = (sum(row[name] * weight for row, weight in zip(available, weights)) / sum(weights)
                        if available else None)
    return output

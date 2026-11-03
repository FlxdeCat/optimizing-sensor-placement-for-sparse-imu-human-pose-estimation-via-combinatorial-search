"""Independent NumPy/SciPy readers for raw DIP and real TotalCapture files.

No dip18 source is copied or imported. Parsing follows its documented numeric
formats; inspection is allowed before approval, real sequence preparation is
gated. Pickles must come from the trusted licensed dataset releases.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
import re
from typing import Any, Iterator

import numpy as np
from scipy.spatial.transform import Rotation

from config import BASELINES, PARENTS, joint_set, SETTINGS, CANDIDATE_POOL
from .data import discover_amass, inspect_amass, load_smpl_payload, pack_selected, verify_vendor
from .storage import atomic_json, sha256_file
import mapping

LOGGER = logging.getLogger("imuplace")


class UnusableSequence(ValueError):
    """Explicitly logged exclusion for residual nonfinite data or too few frames."""


def array_report(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    report: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value.dtype.kind in "fiu":
        finite = np.isfinite(value)
        valid = value[finite]
        report.update(nan_count=int(np.isnan(value).sum()), inf_count=int(np.isinf(value).sum()),
                      finite_min=float(valid.min()) if valid.size else None,
                      finite_max=float(valid.max()) if valid.size else None)
    return report


def orientation_report(ori: np.ndarray) -> dict[str, Any]:
    if ori.shape[-2:] != (3, 3):
        raise ValueError("Orientation array must end in [3,3].")
    matrices = ori.reshape(-1, 3, 3)
    good = matrices[np.isfinite(matrices).all(axis=(1, 2))]
    if not len(good):
        return {"valid_matrices": 0, "orthogonality_max": None, "determinant_min": None, "determinant_max": None}
    gram = good.transpose(0, 2, 1) @ good
    determinants = np.linalg.det(good)
    return {"valid_matrices": len(good), "orthogonality_max": float(np.abs(gram - np.eye(3)).max()),
            "determinant_min": float(determinants.min()), "determinant_max": float(determinants.max())}


def check_rotations(ori: np.ndarray, *, label: str, tolerance: float = 1e-3) -> None:
    report = orientation_report(ori)
    if report["valid_matrices"] != ori.size // 9 or report["orthogonality_max"] > tolerance or abs(report["determinant_min"] - 1) > tolerance or abs(report["determinant_max"] - 1) > tolerance:
        raise ValueError(f"{label}: invalid rotation matrices {report}. Check quaternion order, calibration and missing-data handling.")


def axis_angles_to_local(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    if pose.ndim == 2 and pose.shape[1] == 156:
        body = pose.reshape(-1, 52, 3).copy()
        body[:, 23] = body[:, 37]
        pose = body[:, :24]
    elif pose.ndim == 2 and pose.shape[1] == 72:
        pose = pose.reshape(-1, 24, 3)
    if pose.ndim != 3 or pose.shape[1:] != (24, 3) or pose.dtype.kind not in "fiu" or not np.isfinite(pose).all():
        raise ValueError(f"Expected finite SMPL local axis angles [T,72], [T,24,3] or SMPL+H [T,156]; received {pose.shape}.")
    return Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3).astype(np.float32)


def global_rotations(local: np.ndarray) -> np.ndarray:
    if local.ndim != 4 or local.shape[1:] != (24, 3, 3):
        raise ValueError("Global FK rotations require [T,24,3,3] local matrices.")
    output = np.empty_like(local)
    output[:, 0] = local[:, 0]
    for joint in range(1, 24):
        output[:, joint] = output[:, PARENTS[joint]] @ local[:, joint]
    return output


def read_dip_raw(path: Path) -> dict[str, np.ndarray]:
    try:
        with Path(path).open("rb") as stream:
            value = pickle.load(stream, encoding="latin1")
    except (EOFError, pickle.UnpicklingError, ImportError) as exc:
        raise RuntimeError(f"Cannot decode raw DIP pickle {path}: {exc}. Restore a complete official raw sequence and install the pinned dependencies.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: raw DIP sequence must contain a dictionary.")
    missing = [key for key in ("gt", "imu_acc", "imu_ori") if key not in value]
    if missing:
        raise ValueError(f"{path}: missing raw DIP fields {missing}; use DIP_IMU.zip, not six-sensor normalized downloads.")
    result = {key: np.asarray(value[key]) for key in value if key in ("gt", "imu_acc", "imu_ori", "imu", "sip", "sop")}
    if result["gt"].ndim != 2 or result["gt"].shape[1] != 72:
        raise ValueError(f"{path}: gt must have shape [T,72]; received {result['gt'].shape}.")
    frames = len(result["gt"])
    shapes = {"gt": (frames, 72), "imu_acc": (frames, 17, 3), "imu_ori": (frames, 17, 3, 3),
              "imu": (frames, 17, 12), "sip": (frames, 72), "sop": (frames, 72)}
    for key, array in result.items():
        if array.shape != shapes[key] or array.dtype.kind != "f":
            raise ValueError(f"{path}: {key} is {array.shape}/{array.dtype}, expected floating {shapes[key]}.")
    return result


def infer_dip_layout(data: dict[str, np.ndarray]) -> str:
    if "imu" not in data:
        return "combined_imu_absent; separate imu_acc and imu_ori validated"
    combined, acc, ori = data["imu"], data["imu_acc"], data["imu_ori"].reshape(-1, 17, 9)
    orientation_first = np.allclose(combined[..., :9], ori, equal_nan=True, atol=1e-7, rtol=1e-7) and np.allclose(combined[..., 9:], acc, equal_nan=True, atol=1e-7, rtol=1e-7)
    acceleration_first = np.allclose(combined[..., :3], acc, equal_nan=True, atol=1e-7, rtol=1e-7) and np.allclose(combined[..., 3:], ori, equal_nan=True, atol=1e-7, rtol=1e-7)
    if orientation_first and acceleration_first:
        return "ambiguous: both layouts match these numeric values"
    if orientation_first:
        return "9_orientation_then_3_acceleration"
    if acceleration_first:
        return "3_acceleration_then_9_orientation"
    raise ValueError("Raw DIP imu does not match either packing of imu_ori and imu_acc. Inspect the release or file corruption.")


def fill_nan_neighbours(array: np.ndarray, passes: int = 4) -> np.ndarray:
    """Four simultaneous one-frame forward/backward passes, without smoothing."""
    result = np.array(array, copy=True)
    for _ in range(passes):
        forward = result[:-1].copy()
        result[1:] = np.where(np.isnan(result[1:]), forward, result[1:])
        backward = result[1:].copy()
        result[:-1] = np.where(np.isnan(result[:-1]), backward, result[:-1])
    return result


def dip_orientation_diagnostic(data: dict[str, np.ndarray], max_frames: int = 512) -> dict[str, Any]:
    """Describe matrix-vs-transpose agreement with proxy GT, never choose a fit."""
    clean_frames = np.flatnonzero(np.isfinite(data["gt"]).all(axis=1))
    if not len(clean_frames):
        return {"warning": "No finite proxy GT frames for an orientation diagnostic."}
    indices = clean_frames[np.linspace(0, len(clean_frames) - 1, min(max_frames, len(clean_frames)), dtype=int)]
    target = global_rotations(axis_angles_to_local(data["gt"][indices])).astype(np.float64)
    result = {}
    for joint in (0, 1, 2, 4, 5, 15, 16, 17, 18, 19):
        sensor = mapping.DIP_MAPPING[joint].sensor
        observed = data["imu_ori"][indices, sensor]
        good = np.isfinite(observed).all(axis=(1, 2))
        if not good.any():
            continue
        ground = target[good, joint]
        def mean_angle(matrices):
            relative = ground.transpose(0, 2, 1) @ matrices
            cosine = (np.trace(relative, axis1=1, axis2=2) - 1) / 2
            return float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))).mean())
        result[str(joint)] = {"sensor": sensor, "as_stored_deg": mean_angle(observed[good]),
                              "transposed_deg": mean_angle(observed[good].transpose(0, 2, 1))}
    return {"per_joint": result, "interpretation": "Diagnostic against SIP-derived proxy, not proof of physical units or sensor-to-bone calibration; no transform fitted."}


def inspect_dip(path: Path) -> dict[str, Any]:
    data = read_dip_raw(path)
    finite_matrices = data["imu_ori"][np.isfinite(data["imu_ori"]).all(axis=(2, 3))]
    if len(finite_matrices):
        check_rotations(finite_matrices, label=str(path))
    report = {"file": str(path), "arrays": {key: array_report(value) for key, value in data.items()},
              "combined_layout": infer_dip_layout(data), "orientation_quality": orientation_report(data["imu_ori"]),
              "orientation_diagnostic": dip_orientation_diagnostic(data),
              "sensor_order_source": "supplied overview_of_published_data.txt; index 14 is provisionally rwrist",
              "units_and_frame_status": "require review; numeric ranges alone cannot establish them"}
    for key in ("imu_acc", "imu_ori"):
        bad = ~np.isfinite(data[key]).all(axis=tuple(range(2, data[key].ndim)))
        report[key + "_bad_frames_per_sensor"] = bad.sum(axis=0).tolist()
    filled = {key: fill_nan_neighbours(data[key], SETTINGS.dip_fill_passes)[SETTINGS.dip_trim_frames:-SETTINGS.dip_trim_frames] for key in ("imu_acc", "imu_ori")}
    required=[mapping.DIP_MAPPING[j].sensor for j in CANDIDATE_POOL]
    report["required_stream_indices"]=required
    report["all13_clean_after_configured_preprocessing"] = bool(
        len(data["gt"]) > 2*SETTINGS.dip_trim_frames and all(np.isfinite(value[:,required]).all() for value in filled.values())
        and np.isfinite(data["gt"][SETTINGS.dip_trim_frames:-SETTINGS.dip_trim_frames]).all())
    return report


def quaternion_matrices(values: np.ndarray, *, order: str, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 4 or not np.isfinite(values).all():
        raise ValueError(f"{label}: quaternions must have four finite values.")
    if np.any(np.linalg.norm(values, axis=-1) < 1e-12):
        raise ValueError(f"{label}: zero quaternion cannot define an orientation.")
    if order not in ("wxyz", "xyzw"):
        raise ValueError("Quaternion order must be wxyz or xyzw.")
    reordered = values[..., [1, 2, 3, 0]] if order == "wxyz" else values
    return Rotation.from_quat(reordered.reshape(-1, 4)).as_matrix().reshape(*values.shape[:-1], 3, 3)


def read_tc_raw(path: Path) -> dict[str, Any]:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty TotalCapture .sensors file.")
    # Format from read_TC_data.py: one file header, then a frame delimiter and
    # thirteen labelled rows, each with four quaternion and three acc values.
    body = [line.strip() for line in lines[1:] if line.strip()]
    if not body or len(body) % 14:
        raise ValueError(f"{path}: expected header plus 14 nonempty lines per frame (delimiter + 13 sensors); found {len(body)}. Inspect the file format.")
    frames, labels, delimiters = [], None, []
    for start in range(0, len(body), 14):
        delimiter = body[start]
        if len(delimiter.split()) == 8:
            raise ValueError(f"{path}: expected a frame delimiter before sensor rows at line {start+2}.")
        names, rows = [], []
        for row in body[start + 1:start + 14]:
            tokens = row.split()
            if len(tokens) != 8:
                raise ValueError(f"{path}: sensor row must be label + quaternion4 + acceleration3; received {row!r}.")
            names.append(tokens[0])
            try:
                rows.append([float(token) for token in tokens[1:]])
            except ValueError as exc:
                raise ValueError(f"{path}: nonnumeric sensor values in {row!r}.") from exc
        if len(set(names)) != 13:
            raise ValueError(f"{path}: sensor labels must be thirteen distinct values; found {names}.")
        if labels is None:
            labels = tuple(names)
        elif tuple(names) != labels:
            raise ValueError(f"{path}: sensor row order changes between frames.")
        frames.append(rows)
        if len(delimiters) < 3:
            delimiters.append(delimiter)
    values = np.asarray(frames, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: real IMU rows contain NaN/Inf; they cannot be used without a documented repair protocol.")
    quaternions = values[..., :4]
    return {"labels": labels, "ori": quaternion_matrices(quaternions, order="wxyz", label=str(path)),
            "acc": values[..., 4:], "header": lines[0], "delimiter_examples": delimiters,
            "quaternion_norm_max_deviation": float(np.abs(np.linalg.norm(quaternions, axis=-1) - 1).max())}


def read_tc_calibration(path: Path, sensor_labels: tuple[str, ...]) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"Missing TotalCapture calibration file {path}. Download the corresponding real IMU calibration files.")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    rows = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 5 or tokens[0] in rows:
            raise ValueError(f"{path}: calibration expects distinct label + xyzw quaternion4 rows; received {line!r}.")
        rows[tokens[0]] = [float(value) for value in tokens[1:]]
    if set(rows) != set(sensor_labels):
        raise ValueError(f"{path}: calibration labels {list(rows)} differ from .sensors labels {list(sensor_labels)}.")
    values = np.asarray([rows[label] for label in sensor_labels])
    return quaternion_matrices(values, order="xyzw", label=str(path))


def canonical_tc_order(labels: tuple[str, ...], *, allow_numeric: bool = False) -> tuple[list[int], str]:
    normalized = [re.sub(r"[^a-z0-9]", "", label.lower()) for label in labels]
    if len(normalized) != 13:
        raise ValueError("TotalCapture needs thirteen sensor labels.")
    if all(label.isdigit() for label in normalized):
        if not allow_numeric:
            raise ValueError(f"Numeric labels {labels} do not establish anatomical order. Review the positional mapping before approval.")
        return list(range(13)), "positional draft order independently confirmed through tc_allow_numeric_order"
    indices = [mapping.TC_LABEL_ALIASES.get(label) for label in normalized]
    if any(index is None for index in indices) or set(indices) != set(range(13)):
        raise ValueError(f"Unknown/duplicate anatomical TotalCapture labels {labels}. Add verified label aliases in mapping.py; do not guess an order.")
    return [indices.index(index) for index in range(13)], "anatomical labels matched and reordered to canonical streams"


def tc_sequence_id(path: Path) -> tuple[str, str]:
    match = re.fullmatch(r"(s[1-5])_([a-z]+[0-9]+)_xsens", path.stem.lower())
    if not match:
        raise ValueError(f"Cannot identify TotalCapture subject/motion from {path.name}; expected s1_acting1_Xsens.sensors, etc.")
    return match.group(1), match.group(2)


def discover_tc(root: Path) -> list[tuple[str, str, Path]]:
    files = sorted(root.rglob("*.sensors")) if root.is_dir() else []
    if not files:
        raise RuntimeError(f"No real TotalCapture .sensors files below {root}. Request the real IMU and calibration data; see README.md section 3.")
    result, seen = [], set()
    for file in files:
        subject, motion = tc_sequence_id(file)
        if (subject, motion) in mapping.TC_EXCLUDED:
            continue
        if (subject, motion) in seen:
            raise ValueError(f"Duplicate TotalCapture sequence {subject}/{motion} under {root}. Keep one source copy.")
        seen.add((subject, motion))
        result.append((subject, motion, file))
    return sorted(result)


def calibration_paths(imu: Path) -> tuple[Path, Path]:
    suffix = "_xsens"
    if not imu.stem.lower().endswith(suffix):
        raise ValueError(f"{imu}: missing _Xsens filename suffix.")
    stem = imu.stem[:-len(suffix)]
    expected = [imu.with_name(stem + ending) for ending in ("_calib_imu_ref.txt", "_calib_imu_bone.txt")]
    # Case-insensitive discovery also works when archives are inspected on Linux.
    children = {path.name.lower(): path for path in imu.parent.iterdir()}
    return tuple(children.get(path.name.lower(), path) for path in expected)


def find_reference(root: Path, subject: str, motion: str) -> Path:
    candidates = []
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in ("", ".pkl", ".npz"):
                continue
            direct = path.parent.name.lower() == subject and path.stem.lower() == motion
            flat = path.stem.lower() == f"{subject}_{motion}"
            if direct or flat:
                candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"Need exactly one SMPL reference for {subject}/{motion} under {root}; found {candidates}. "
                           f"Place the matching poses at {root / (subject + '_' + motion + '.pkl')} (nested subject folders or NPZ also accepted).")
    return candidates[0]


def reference_inventory(root: Path) -> set[tuple[str, str]]:
    """Filename inventory can expose missing real streams when poses are present."""
    result = set()
    for path in root.rglob("*") if root.is_dir() else ():
        if not path.is_file() or path.suffix.lower() not in ("", ".pkl", ".npz"):
            continue
        subject, motion = path.parent.name.lower(), path.stem.lower()
        flat = re.fullmatch(r"(s[1-5])_([a-z]+[0-9]+)", motion)
        if flat:
            subject, motion = flat.groups()
        if subject in mapping.TC_TEST_SUBJECTS and re.fullmatch(r"[a-z]+[0-9]+", motion):
            if (subject, motion) not in mapping.TC_EXCLUDED:
                result.add((subject, motion))
    return result


def read_tc_reference(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
    else:
        try:
            with path.open("rb") as stream:
                data = pickle.load(stream, encoding="latin1")
        except (EOFError, pickle.UnpicklingError, ImportError) as exc:
            raise RuntimeError(f"Cannot decode TotalCapture reference {path}: {exc}. Restore the original licensed pose file.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: TotalCapture reference must contain a dictionary.")
    if "poses" in data:
        key, default_offset, source_type = "poses", 0, "native MoSh poses"
    elif "gt" in data:
        bundled = "acc" in data and "ori" in data
        if bundled:
            unused_acc, unused_ori = np.asarray(data["acc"]), np.asarray(data["ori"])
            if unused_acc.ndim != 3 or unused_acc.shape[1:] != (6, 3) or unused_ori.shape != (len(unused_acc), 6, 3, 3):
                raise ValueError(f"{path}: expected original DIP TotalCapture bundle with acc[T,6,3], ori[T,6,3,3]. "
                                 "Other processed bundles require an explicit pose-only export with reference_frame_offset metadata.")
        elif "reference_frame_offset" not in data:
            raise ValueError(f"{path}: standalone 'gt' has ambiguous time origin. Include integer reference_frame_offset "
                             "from its documented producer, or provide native 'poses'. Never infer an offset from test errors.")
        # The audited dip18 writer stores mosh_gt[1:]. Its six-sensor signals
        # are ignored; they are NOT training or evaluation features here.
        key, default_offset, source_type = "gt", 1, "DIP original TotalCapture pose reference; six-stream inputs ignored"
    else:
        raise ValueError(f"{path}: missing 'poses' or documented TotalCapture 'gt' reference field.")
    offset_array = np.asarray(data.get("reference_frame_offset", default_offset))
    if offset_array.size != 1 or offset_array.dtype.kind not in "iu" or int(offset_array.reshape(-1)[0]) < 0:
        raise ValueError(f"{path}: reference_frame_offset must be one non-negative integer.")
    offset = int(offset_array.reshape(-1)[0])
    if "mocap_framerate" in data and abs(float(np.asarray(data["mocap_framerate"]).reshape(-1)[0]) - 60) > 1e-6:
        raise ValueError(f"{path}: TotalCapture reference must be synchronized to 60 Hz; found {data['mocap_framerate']}.")
    pose = axis_angles_to_local(np.asarray(data[key]))
    if len(pose) < 2:
        raise ValueError(f"{path}: reference needs at least two frames for known-pose calibration.")
    return pose, {"file": str(path), "poses_input_shape": list(np.asarray(data[key]).shape),
                  "pose_field": key, "raw_frame_offset": offset, "producer": source_type,
                  "frames": len(pose), "reference_type": "SMPL fit to optical mocap; not raw Vicon marker positions"}


def find_vicon(root: Path, subject: str, motion: str) -> Path:
    matches = [path for path in root.rglob("*") if path.is_file()
               and path.name.lower() == "gt_skel_gbl_pos.txt"
               and path.parent.name.lower() == motion and path.parent.parent.name.lower() == subject] if root.is_dir() else []
    if len(matches) != 1:
        raise RuntimeError(f"Need one Vicon gt_skel_gbl_pos.txt for {subject}/{motion}; found {matches}. "
                           f"Extract Vicon data preserving {root / subject.upper() / motion / 'gt_skel_gbl_pos.txt'}.")
    return matches[0]


def read_vicon(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: missing Vicon header or position frames.")
    header = lines[0].rstrip("\t").split("\t")
    has_frame = header[0].strip().lower() in ("frame", "frameid", "time")
    names = tuple(value.strip() for value in (header[1:] if has_frame else header))
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{path}: expected distinct tab-separated Vicon joint labels.")
    positions = []
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        columns = line.rstrip("\t").split("\t")
        if has_frame:
            columns = columns[1:]
        if len(columns) != len(names):
            raise ValueError(f"{path}:{number}: {len(columns)} Vicon columns, expected {len(names)}.")
        row = [[float(value) for value in column.split()] for column in columns]
        if any(len(vector) != 3 for vector in row):
            raise ValueError(f"{path}:{number}: each Vicon joint must contain three coordinates.")
        positions.append(row)
    result = np.asarray(positions, dtype=np.float64)
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError(f"{path}: Vicon coordinates are empty or nonfinite.")
    # Format and conversion documented by TransPose preprocessing, not guessed
    # from coordinate magnitude. Preserve names; they are not SMPL joint indices.
    result *= 0.0254
    result[..., (0, 2)] *= -1
    return names, result.astype(np.float32)


@dataclass
class RealSequence:
    id: str
    dataset: str
    acc: np.ndarray
    ori: np.ndarray
    pose: np.ndarray
    trans: np.ndarray
    diagnostics: dict[str, Any]
    vicon: np.ndarray | None = None
    vicon_names: tuple[str, ...] = ()

    def features(self, joints: Any):
        chosen = joint_set(joints)
        sensors = mapping.sensor_indices(self.dataset, chosen)
        frames = len(self.pose)
        acc = np.zeros((frames, 24, 3), dtype=np.float32)
        ori = np.zeros((frames, 24, 3, 3), dtype=np.float32)
        acc[:, list(chosen)] = self.acc[:, list(sensors)]
        ori[:, list(chosen)] = self.ori[:, list(sensors)]
        LOGGER.info("Real normalization %s joints=%s sensors=%s anchor_joint=%d; upstream selected-anchor /30 and fixed zero mask",
                    self.id, chosen, sensors, chosen[0])
        return pack_selected(acc, ori, chosen)


def load_dip_sequence(path: Path, sequence_id: str) -> RealSequence:
    mapping.require_approval()
    if mapping.DIP_CONVENTIONS["acceleration"] != "world_linear_m_s2" or mapping.DIP_CONVENTIONS["orientation"] != "segment_to_world" or mapping.DIP_CONVENTIONS["fps"] != 60.0:
        raise RuntimeError("DIP convention mismatch: this reader expects world linear m/s^2 and segment-to-world matrices. "
                           "Update the reader with a reviewed conversion before evaluating; normalization must match synthetic training.")
    data = read_dip_raw(path)
    layout = infer_dip_layout(data)
    acc = fill_nan_neighbours(data["imu_acc"], SETTINGS.dip_fill_passes)[SETTINGS.dip_trim_frames:-SETTINGS.dip_trim_frames].astype(np.float32)
    ori = fill_nan_neighbours(data["imu_ori"], SETTINGS.dip_fill_passes)[SETTINGS.dip_trim_frames:-SETTINGS.dip_trim_frames].astype(np.float32)
    pose = data["gt"][SETTINGS.dip_trim_frames:-SETTINGS.dip_trim_frames]
    required=[mapping.DIP_MAPPING[j].sensor for j in CANDIDATE_POOL]
    if not len(pose) or not all(np.isfinite(value).all() for value in (acc[:,required], ori[:,required], pose)):
        raise UnusableSequence(f"{sequence_id}: excluded after {SETTINGS.dip_fill_passes} neighbour passes and trimming {SETTINGS.dip_trim_frames} each end: residual NaN/Inf in the common 13-stream pool/pose or no frames.")
    check_rotations(ori[:,required], label=sequence_id)
    diagnostics = {"source": str(path), "source_sha256": sha256_file(path), "input_layout": layout,
                   "original_frames": len(data["gt"]), "retained_frames": len(pose),
                   "nan_before": {key: int(np.isnan(data[key]).sum()) for key in ("imu_acc", "imu_ori")},
                   "cohort_rule": "all 13 required streams and poses finite after common preprocessing; common cohort for all six-site and all-13 configurations",
                   "required_stream_indices":required,
                   "ground_truth": "SIP-derived 17-sensor proxy, not optical mocap",
                   "convention_confirmation": "declared through mapping approval, not inferred from numeric ranges",
                   "mapping_fingerprint": mapping.mapping_fingerprint()}
    return RealSequence(sequence_id, "dip", acc, ori, axis_angles_to_local(pose),
                        np.zeros((len(pose), 3), dtype=np.float32), diagnostics)


def iter_dip_sequences(root: Path, *, deadline=None) -> Iterator[RealSequence]:
    mapping.require_approval()
    for subject in mapping.DIP_TEST_SUBJECTS:
        files = sorted((root / subject).glob("*.pkl"))
        if not files:
            raise RuntimeError(f"Missing DIP standard test subject {root / subject}. Extract raw DIP_IMU.zip.")
        for file in files:
            if deadline is not None:
                deadline.check()
            try:
                sequence = load_dip_sequence(file, subject + "/" + file.stem)
            except UnusableSequence as exc:
                LOGGER.warning("%s", exc)
                continue
            LOGGER.info("DIP prepared %s: %s", sequence.id, sequence.diagnostics)
            yield sequence


def load_tc_sequence(imu_file: Path, reference_file: Path, vicon_file: Path | None,
                     subject: str, motion: str) -> RealSequence:
    mapping.require_approval()
    expected = {"fps": 60.0, "quaternion_sensors": "wxyz", "quaternion_calibration": "xyzw",
                "acceleration": "sensor_specific_force_m_s2", "world_conversion": "rotate_Y_pi_after_reference_calibration",
                "orientation_calibration": "first_reference_pose_known_pose", "vicon_input_unit": "inches",
                "vicon_world_conversion": "negate_x_and_z"}
    mismatches = {key: mapping.TC_CONVENTIONS.get(key) for key, value in expected.items()
                  if mapping.TC_CONVENTIONS.get(key) != value}
    if mismatches:
        raise RuntimeError(f"TotalCapture preprocessing convention mismatch {mismatches}. Implement a reviewed conversion before evaluation.")
    if not SETTINGS.tc_first_reference_pose_calibration:
        raise RuntimeError("Only the disclosed first-reference-pose calibration is implemented. Enable tc_first_reference_pose_calibration or supply/review a complete calibration-only conversion before TC evaluation.")
    raw = read_tc_raw(imu_file)
    ref_path, bone_path = calibration_paths(imu_file)
    calibration = read_tc_calibration(ref_path, raw["labels"])
    read_tc_calibration(bone_path, raw["labels"])  # verify availability; not used by the reference convention
    order, order_status = canonical_tc_order(raw["labels"], allow_numeric=SETTINGS.tc_allow_numeric_order)
    local, reference_report = read_tc_reference(reference_file)
    vicon_names, vicon_positions = read_vicon(vicon_file) if vicon_file is not None else ((), None)
    if vicon_positions is not None and "Spine" not in vicon_names:
        raise ValueError(f"{vicon_file}: TransPose root-translation convention needs the Vicon Spine label; found {vicon_names}.")
    y_pi = np.diag([-1.0, 1.0, -1.0])
    sensor_to_world = y_pi @ calibration[None] @ raw["ori"]
    physical_acc = np.einsum("tsij,tsj->tsi", sensor_to_world, raw["acc"]) - np.asarray(mapping.TC_CONVENTIONS["gravity_world"])
    sensor_to_world, physical_acc = sensor_to_world[:, order], physical_acc[:, order]
    offset = reference_report["raw_frame_offset"]
    if offset >= len(sensor_to_world):
        raise ValueError(f"{reference_file}: reference frame offset {offset} exceeds the real recording length.")
    # Native canonical streams correspond to these segment rotations. Feet are
    # ankle-defined foot segments; toe joints remain explicit placement proxies.
    bone_joints = np.asarray([15, 9, 0, 16, 17, 18, 19, 1, 2, 4, 5, 7, 8])
    reference_global0 = global_rotations(local[:1])[0, bone_joints].astype(np.float64)
    calibrated_ori = sensor_to_world @ sensor_to_world[offset].transpose(0, 2, 1)[None] @ reference_global0[None]
    counts = {"imu": len(sensor_to_world), "reference": len(local), "vicon": len(vicon_positions) if vicon_positions is not None else len(sensor_to_world)}
    aligned_counts = {"imu": counts["imu"] - offset, "reference": counts["reference"], "vicon": counts["vicon"] - offset}
    frames = min(aligned_counts["imu"],aligned_counts["reference"])
    if vicon_positions is not None and len(vicon_positions) < offset + frames:
        LOGGER.warning("Optional Vicon shorter than the IMU/SMPL cohort; omit its diagnostic array without shortening SMPL evaluation.")
        vicon_positions = None; vicon_names = ()
    if frames < 5:
        raise UnusableSequence(f"{subject}/{motion}: too few synchronized frames after known-pose calibration.")
    if len(set(aligned_counts.values())) > 1:
        LOGGER.warning("TotalCapture length mismatch %s/%s: %s, reference offset=%d, aligned=%s; truncate to %d.", subject, motion, counts, offset, aligned_counts, frames)
    retained = slice(offset + 1, offset + frames)
    retained_reference = slice(1, frames)
    if vicon_positions is not None:
        root = vicon_positions[:, vicon_names.index("Spine")]
        translation = root - root[:1]
    else:
        translation = np.zeros((len(sensor_to_world),3),dtype=np.float32)
    ori = calibrated_ori[retained].astype(np.float32)
    check_rotations(ori, label=f"{subject}/{motion}")
    diagnostics = {"source": str(imu_file), "sensor_labels": raw["labels"], "order_status": order_status,
                   "quaternion_norm_max_deviation": raw["quaternion_norm_max_deviation"],
                   "source_sha256": sha256_file(imu_file), "reference_sha256": sha256_file(reference_file),
                   "reference_calibration_sha256": sha256_file(ref_path), "bone_calibration_sha256": sha256_file(bone_path),
                   "vicon_sha256": sha256_file(vicon_file) if vicon_file is not None else None, "reference": reference_report,
                   "original_frames": counts, "retained_frames": frames - 1,
                   "aligned_frames": aligned_counts, "reference_frame_offset": offset,
                   "calibration_raw_frame": offset, "first_evaluated_raw_frame": offset + 1,
                   "calibration": "uses first available ground-truth SMPL pose; excludes that frame and any earlier unpaired raw frames",
                   "bone_calibration": "read and validated; unused, matching dip18 reference convention",
                   "gravity_world": mapping.TC_CONVENTIONS["gravity_world"],
                   "vicon_unit_conversion": "inches to meters, x/z negated",
                   "vicon_use": "named optical positions retained separately; SMPL metrics use the SMPL reference, not index-matched Vicon joints",
                   "mapping_fingerprint": mapping.mapping_fingerprint()}
    if raw["quaternion_norm_max_deviation"] > 1e-3:
        LOGGER.warning("%s: non-unit input quaternions normalized by SciPy; maximum norm deviation=%g", imu_file, raw["quaternion_norm_max_deviation"])
    LOGGER.warning("TotalCapture %s/%s uses a known first GT pose for orientation calibration; this is disclosed calibration information.", subject, motion)
    return RealSequence(subject + "/" + motion, "totalcapture", physical_acc[retained].astype(np.float32),
                        ori, local[retained_reference], translation[retained], diagnostics,
                        None if vicon_positions is None else vicon_positions[retained], vicon_names)


def iter_tc_sequences(root: Path, reference_root: Path, *, deadline=None) -> Iterator[RealSequence]:
    mapping.require_approval()
    for subject, motion, imu_file in discover_tc(root):
        if deadline is not None:
            deadline.check()
        from .real_evaluation import optional_vicon
        if subject not in SETTINGS.tc_subjects:continue
        sequence = load_tc_sequence(imu_file, find_reference(reference_root, subject, motion),
                                    optional_vicon(root, subject, motion), subject, motion)
        LOGGER.info("TotalCapture prepared %s: %s", sequence.id, sequence.diagnostics)
        yield sequence


"""Physical AMASS cache and fixed-pelvis input packing. No search code lives here."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import json, logging, math, pickle, re, shutil, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from config import IMUOPTIMIZE_COMMIT,PARENTS,PIPELINE_VERSION,Paths,Settings,TrainingBudget,fingerprint,joint_set
from .storage import atomic_json,atomic_write,sha256_file
from .vendor_adapter import VendorAdapter,vendor,verify_vendor,load_smpl_payload
_LOGGER=logging.getLogger("imuplace")
def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run 0_check.py and inspect nvidia-smi.")
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else
                        "cpu" if requested == "auto" else requested)


def discover_amass(root: Path) -> list[Path]:
    files = sorted(root.rglob("*_poses.npz")) if root.is_dir() else []
    if not files:
        raise RuntimeError(f"No AMASS CMU *_poses.npz files below {root}. Extract the CMU SMPL+H archive; see README.md section 3.")
    return files


def amass_subject(path: Path) -> str:
    """Return a conservative CMU split group, preserving original paths.

    Known paired Justin/Rory folders use shared motion prefixes. Group all
    releases 18--23 together rather than guessing a person from that prefix.
    Numeric folders 18--23 join that group if also present. Other numeric
    release IDs remain their own groups; biological identities are not proven.
    """
    parent = path.parent.name
    prefix = path.name.split("_")[0]
    if not prefix.isdigit():
        raise ValueError(f"Cannot identify a CMU motion prefix in {path.name}. "
                         "Preserve the original *_poses.npz filenames.")
    number = int(prefix)
    if re.fullmatch(r"[0-9]+", parent):
        subject = int(parent)
        if number != subject:
            raise ValueError(f"CMU subject mismatch: filename {path.name} versus folder {parent}.")
        return "paired_18_23" if 18 <= subject <= 23 else str(subject).zfill(2)
    paired = re.fullmatch(r"(18_19|20_21|22_23)_(justin|rory)(1)?", parent, flags=re.IGNORECASE)
    if paired:
        ids = tuple(int(value) for value in paired.group(1).split("_"))
        if number not in ids:
            raise ValueError(f"CMU paired motion mismatch: {path.name} is outside {ids} in {parent}.")
        return "paired_18_23"
    raise ValueError(f"Cannot establish a CMU split group for {path}. Keep original numeric or "
                     "known 18_19/20_21/22_23 Justin/Rory folders. For another named folder, "
                     "supply its original name and recording metadata before adding a mapping.")


def inspect_amass(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("poses", "trans", "betas", "mocap_framerate")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"{path}: missing AMASS fields {missing}; use the SMPL+H motion archive.")
        pose, trans, betas = archive["poses"], archive["trans"], archive["betas"]
        rate_array = np.asarray(archive["mocap_framerate"])
        if rate_array.size != 1 or not np.isfinite(rate_array).all():
            raise ValueError(f"{path}: mocap_framerate must be one finite value.")
        rate = float(rate_array.reshape(-1)[0])
        if pose.ndim != 2 or pose.shape[1] != 156 or trans.shape != (len(pose), 3):
            raise ValueError(f"{path}: poses {pose.shape}, trans {trans.shape}; required [T,156] SMPL+H and [T,3].")
        if betas.ndim != 1 or len(betas) < 10:
            raise ValueError(f"{path}: betas must be a vector with at least ten entries.")
        if any(array.dtype.kind not in "fiu" or not np.isfinite(array).all() for array in (pose, trans, betas)):
            raise ValueError(f"{path}: AMASS poses, trans and betas must be finite numeric arrays.")
        step = 2 if int(rate) == 120 else 1
        frames = (len(pose) + step - 1) // step
        supported = int(rate) in (59, 60, 120) and frames > 12
        reason = None if supported else "upstream rejects this frame rate or sequences with <=12 retained frames"
        return {"subject": amass_subject(path), "source_subject_folder": path.parent.name,
                "subject_identity_status": "conservative_archive_split_group_not_verified_person",
                "poses": list(pose.shape),
                "trans": list(trans.shape), "betas": list(betas.shape), "source_fps": rate,
                "retained_frames": frames, "stride": step, "effective_fps": rate / step,
                "synthesis_fps": 60.0, "supported": supported, "skip_reason": reason}


def subject_split(subjects: list[str], preset: Settings) -> dict[str, list[str]]:
    ordered = np.asarray(sorted(set(subjects)))
    if len(ordered) < 2:
        raise ValueError("AMASS needs at least two distinct subjects for a subject-disjoint train/validation split.")
    shuffled = np.random.default_rng(preset.split_seed).permutation(ordered).tolist()
    count = min(len(shuffled) - 1, max(1, int(math.floor(len(shuffled) * preset.train_subject_fraction))))
    train, val = shuffled[:count], shuffled[count:]
    if preset.max_train_subjects is not None:
        train = train[:preset.max_train_subjects]
    if preset.max_val_subjects is not None:
        val = val[:preset.max_val_subjects]
    return {"train": sorted(train), "validation": sorted(val)}


@torch.no_grad()
def synthesize_sequence(adapter: VendorAdapter, body, sequence, *, chunk_frames: int = 64,
                        smooth_n: int = 4, deadline=None, progress_dir: Path | None = None) -> dict[str, np.ndarray]:
    if chunk_frames < 1 or smooth_n != 4:
        raise ValueError("Use a positive FK chunk size and the agreed upstream smooth_n=4.")
    pose, trans, betas = sequence
    if len(pose) <= 12 or pose.shape[1:] != (156,):
        raise ValueError("Synthesis requires more than twelve frames and poses [T,156].")
    output: dict[str, list[np.ndarray]] = {name: [] for name in ("pose", "trans", "joint", "site_pos", "acc", "ori")}
    sites = list(range(24))
    last_log=time.monotonic()
    for start in range(0, len(pose), chunk_frames):
        if deadline is not None:
            deadline.check()
        end = min(len(pose), start + chunk_frames)
        partial = progress_dir / f"{start:09d}.npz" if progress_dir is not None else None
        meta = partial.with_suffix(".json") if partial is not None else None
        if partial is not None and partial.is_file() and meta.is_file():
            metadata = json.loads(meta.read_text(encoding="utf-8"))
            if metadata.get("sha256") == sha256_file(partial):
                with np.load(partial, allow_pickle=False) as archive:
                    saved = {key: archive[key] for key in archive.files}
                expected = {"pose": (end-start, 24, 3, 3), "ori": (end-start, 24, 3, 3),
                            "acc": (end-start, 24, 3), "site_pos": (end-start, 24, 3),
                            "joint": (end-start, 24, 3), "trans": (end-start, 3)}
                if all(key in saved and saved[key].shape == shape and np.isfinite(saved[key]).all()
                       for key, shape in expected.items()):
                    for name in output:
                        output[name].append(saved[name])
                    continue
        left, right = max(0, start - smooth_n), min(len(pose), end + smooth_n)
        if right - left <= 12:
            left = max(0, right - 13)
            right = min(len(pose), max(right, left + 13))
        local_pose, local_trans = adapter.pipeline.transform_poses_trans(
            pose[left:right].clone().to(adapter.device), trans[left:right].clone().to(adapter.device))
        global_ori, joints, mesh = adapter.pipeline.run_kinematics(
            body, local_pose, local_trans, betas.to(adapter.device))
        norm_acc, _ = adapter.pipeline.synthesize_imu_data(body, mesh, global_ori, sites, smooth_n)
        root = global_ori[:, 0]
        # Invert normalization: root acceleration was NOT root-subtracted.
        delta = torch.bmm(norm_acc * 30.0, root.transpose(1, 2))
        physical_acc = delta.clone()
        physical_acc[:, 1:] += delta[:, :1]
        center = slice(start - left, end - left)
        values = {"pose": local_pose, "trans": local_trans, "joint": joints,
                  "site_pos": mesh[:, body._J_regressor.argmax(dim=1)],
                  "acc": physical_acc, "ori": global_ori}
        chunk = {name: value[center].detach().cpu().numpy().astype(np.float32, copy=True)
                 for name, value in values.items()}
        for name, value in chunk.items():
            output[name].append(value)
        if partial is not None:
            atomic_write(partial, lambda stream: np.savez_compressed(stream, **chunk))
            atomic_json(meta, {"sha256": sha256_file(partial), "start": start, "end": end})
        del mesh, norm_acc, global_ori, values
        if time.monotonic()-last_log>=60:
            _LOGGER.info("SYNTHESIS FK progress %d/%d frames",end,len(pose));last_log=time.monotonic()
    result = {name: np.concatenate(parts) for name, parts in output.items()}
    result["betas"] = betas.cpu().numpy().astype(np.float32)
    return result


def pack_selected(acc: np.ndarray | torch.Tensor, ori: np.ndarray | torch.Tensor,
                  joints: Any, *, adapter: VendorAdapter | None = None) -> torch.Tensor:
    """Normalize only selected physical streams, then zero-mask fixed slots.

    Input [T,24,3] and [T,24,3,3]; output [T,288]. Channel order is 72
    accelerations followed by 216 orientations, matching the upstream loader.
    """
    chosen = joint_set(joints)
    acc = torch.as_tensor(acc, dtype=torch.float32)
    ori = torch.as_tensor(ori, dtype=torch.float32, device=acc.device)
    if acc.ndim != 3 or acc.shape[1:] != (24, 3) or ori.shape != (*acc.shape[:2], 3, 3):
        raise ValueError(f"Physical cache shapes invalid: acc {tuple(acc.shape)}, ori {tuple(ori.shape)}.")
    if chosen[0] != 0:
        raise ValueError("Pelvis joint 0 is mandatory for every model input.")
    selection = list(chosen)
    selected_acc, selected_ori = acc[:, selection].contiguous(), ori[:, selection].contiguous()
    if not torch.isfinite(selected_acc).all() or not torch.isfinite(selected_ori).all():
        raise ValueError("Selected physical signals contain NaN/Inf; inspect E0 and real preprocessing logs.")
    adapter = adapter or vendor("cpu")
    norm_acc, norm_ori = adapter.pipeline.normalize_imu_data(selected_acc, selected_ori, len(chosen))
    masked_acc, masked_ori = torch.zeros_like(acc), torch.zeros_like(ori)
    masked_acc[:, selection], masked_ori[:, selection] = norm_acc, norm_ori
    return torch.cat((masked_acc.flatten(1), masked_ori.flatten(1)), dim=1)


def _source_hashes(files: list[Path], root: Path, target: Path, deadline) -> dict[str, str]:
    previous = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    current = {}
    for path in files:
        deadline.check()
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        entry = previous.get(relative)
        stamp = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if not entry or entry.get("stamp") != stamp:
            entry = {"stamp": stamp, "sha256": sha256_file(path)}
        current[relative] = entry
        # Preserve not-yet-visited rows to make the hashing phase resumable.
        previous[relative] = entry
        atomic_json(target, previous)
    atomic_json(target, current)
    return {name: entry["sha256"] for name, entry in current.items()}




def load_index(paths: Paths, *, verify_hashes: bool = False) -> dict[str, Any]:
    file = paths.cache / "index.json"
    if not file.is_file():
        raise RuntimeError(f"Missing synthetic cache {file}. Run 1_synthesize.py with the same config.py settings first.")
    index = json.loads(file.read_text(encoding="utf-8"))
    if index.get("schema") != 1 or index.get("status") != "complete":
        raise RuntimeError(f"Synthetic cache is incomplete at {file}. Resume 1_synthesize.py before training.")
    if fingerprint(index["settings"]) != index["data_fingerprint"]:
        raise RuntimeError("Cache settings fingerprint is corrupt. Restore the cache manifest or regenerate in a new root.")
    if sha256_file(Path(__file__)) != index["settings"]["adapter_sha256"]:
        raise RuntimeError("Synthesis adapter changed since this cache was generated. Rerun 1_synthesize.py with the current config; changed synthesis settings select a new cache namespace.")
    if not paths.smpl.is_file() or sha256_file(paths.smpl) != index["settings"]["smpl_sha256"]:
        raise RuntimeError("SMPL model is missing or changed since synthesis. Restore that model or rerun 1_synthesize.py after reviewing the configured body model.")
    for row in index["sequences"]:
        path = (paths.cache / row["cache_file"]).resolve()
        if not path.is_relative_to(paths.cache.resolve()) or not path.is_file():
            raise RuntimeError(f"Missing or invalid cache file {path}. Resume 1_synthesize.py to repair missing files.")
        if verify_hashes and sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Cache hash mismatch at {path}. Resume 1_synthesize.py to regenerate it from source.")
    return index


def load_cached_sequence(paths: Paths, row: dict[str, Any]) -> dict[str, np.ndarray]:
    file = paths.cache / row["cache_file"]
    with np.load(file, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    frames = row["frames"]
    shapes = {"pose": (frames, 24, 3, 3), "acc": (frames, 24, 3),
              "ori": (frames, 24, 3, 3), "joint": (frames, 24, 3),
              "site_pos": (frames, 24, 3),
              "trans": (frames, 3), "betas": (10,)}
    for key, expected in shapes.items():
        if key not in data or data[key].shape != expected or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} in {file}; expected finite {expected}. Resume 1_synthesize.py to rebuild after restoring its manifest hash.")
    return data


def windows_for_length(frames: int, window: int, stride: int) -> list[tuple[int, int]]:
    if frames < 1 or window < 1 or not 1 <= stride <= window:
        raise ValueError("Require positive frame/window counts and 1 <= stride <= window.")
    starts = list(range(0, max(1, frames - window + 1), stride))
    if starts[-1] + window < frames:
        starts.append(starts[-1] + stride)
        while starts[-1] + window < frames:
            starts.append(starts[-1] + stride)
    return [(start, min(window, frames - start)) for start in starts]


class AMASSWindowDataset(Dataset):
    """Small CPU sequence cache; fixed windows with explicit valid-frame masks.

    A training-window fraction is drawn once using split_seed, independently of
    architecture, sensor subset, and model seed. Validation always uses all its
    subject-disjoint sequences. Never pad across a sequence boundary.
    """
    def __init__(self, paths: Paths, index: dict[str, Any], joints: Any,
                 budget: TrainingBudget, *, partition: str, split_seed: int):
        if partition not in ("train", "validation"):
            raise ValueError("Partition must be train or validation.")
        self.paths, self.joints, self.budget = paths, joint_set(joints), budget
        self.rows = [row for row in index["sequences"] if row["split"] == partition]
        self.windows = [(i, start, valid) for i, row in enumerate(self.rows)
                        for start, valid in windows_for_length(row["frames"], budget.window_length, budget.stride)]
        if partition == "train" and budget.train_fraction < 1:
            count = max(1, int(math.floor(len(self.windows) * budget.train_fraction)))
            selected = sorted(np.random.default_rng(split_seed).choice(len(self.windows), count, replace=False).tolist())
            self.windows = [self.windows[i] for i in selected]
        if not self.windows:
            raise RuntimeError(f"No {partition} windows. Inspect E1 index.json and subject split.")
        self._sequence_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sequence_index, start, valid_count = self.windows[item]
        if sequence_index not in self._sequence_cache:
            data = load_cached_sequence(self.paths, self.rows[sequence_index])
            self._sequence_cache[sequence_index] = (pack_selected(data["acc"], data["ori"], self.joints),
                                                     torch.from_numpy(data["pose"]).float())
            while len(self._sequence_cache) > 2:
                self._sequence_cache.popitem(last=False)
        self._sequence_cache.move_to_end(sequence_index)
        features, rotations = self._sequence_cache[sequence_index]
        x, y = features[start:start + valid_count], rotations[start:start + valid_count]
        padding = self.budget.window_length - valid_count
        if padding:
            x = torch.cat((x, x[-1:].expand(padding, -1)))
            y = torch.cat((y, y[-1:].expand(padding, -1, -1, -1)))
        return {"x": x, "y": y, "valid": torch.arange(self.budget.window_length) < valid_count,
                "sequence": self.rows[sequence_index]["id"], "start": start}

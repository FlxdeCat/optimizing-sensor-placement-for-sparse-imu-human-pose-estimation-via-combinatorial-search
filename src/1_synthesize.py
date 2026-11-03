"""Synthesize physical 24-site IMUs once, with the configured CMU split."""
from typing import Any
import time,shutil,json
from pathlib import Path
import numpy as np
import torch
from config import PIPELINE_VERSION,IMUOPTIMIZE_COMMIT,fingerprint
from shared import data
from shared.data import choose_device,discover_amass,amass_subject,subject_split,inspect_amass,_source_hashes,synthesize_sequence
from shared.vendor_adapter import load_smpl_payload,vendor
from shared.storage import atomic_json,atomic_write,sha256_file
from shared.session import run_step

def main(context) -> None:
    paths, preset = context.paths, context.settings
    smpl_payload = load_smpl_payload(paths.smpl)  # validate before hashing motion files
    files = discover_amass(paths.amass)
    device = choose_device(context.settings.device)
    chunk_frames = context.settings.fk_chunk_frames
    if chunk_frames < 1:
        raise ValueError("fk_chunk_frames in config.py must be positive.")
    # FK and finite differences are never computed under autocast.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    # Split from preserved subject folders before reading motion arrays. PILOT
    # must not load/hash the entire CMU archive to synthesize three short clips.
    all_subjects = sorted({amass_subject(file) for file in files})
    group_folders: dict[str, list[str]] = {}
    for file in files:
        group = amass_subject(file)
        group_folders.setdefault(group, [])
        if file.parent.name not in group_folders[group]:
            group_folders[group].append(file.parent.name)
    context.logger.info("CMU split groups and preserved folders: %s", group_folders)
    if "paired_18_23" in group_folders:
        context.logger.warning("CMU 18--23 recordings share one conservative split group; "
                               "folder/prefix numbers do not establish biological subject identity.")
    split = subject_split(all_subjects, preset)
    included = set(split["train"] + split["validation"])
    counts: dict[str, int] = {}
    selected, schemas = [], {}
    for file in files:
        context.deadline.check()
        relative = file.relative_to(paths.amass).as_posix()
        subject = amass_subject(file)
        if subject not in included:
            continue
        if preset.max_sequences_per_subject is not None and counts.get(subject, 0) >= preset.max_sequences_per_subject:
            continue
        schema = inspect_amass(file)
        schemas[relative] = schema
        if not schema["supported"]:
            context.logger.warning("Upstream skip %s: %s", relative, schema["skip_reason"])
            continue
        selected.append(relative)
        counts[subject] = counts.get(subject, 0) + 1
    if not selected:
        raise RuntimeError("No supported motions remain after subject split and preset caps. Inspect environment_and_data_check.json.")
    source_hashes = _source_hashes([paths.amass / name for name in selected], paths.amass,
                                 paths.cache / "source_hashes.json", context.deadline)
    settings = {"pipeline_version": PIPELINE_VERSION, "upstream_commit": IMUOPTIMIZE_COMMIT,
                "adapter_sha256": sha256_file(Path(data.__file__)), "smpl_sha256": sha256_file(paths.smpl),
                "source_hashes": source_hashes, "split": split, "all_subjects": all_subjects,
                "split_group_folders": group_folders,
                "split_unit": "numeric_archive_group_with_conservative_18_23_block",
                "sensor_vertex_indices": smpl_payload["J_regressor"].argmax(axis=1).tolist(),
                "body_model": "fixed male SMPL", "use_pose_blendshape": False,
                "selected_files": selected,
                "max_frames": preset.max_frames_per_sequence, "smooth_n": preset.smooth_n,
                "synthesis_fps": preset.fps, "chunk_frames": chunk_frames,
                "device": str(device), "torch": torch.__version__}
    identity = fingerprint(settings)
    index_path = paths.cache / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("data_fingerprint") != identity:
            raise RuntimeError(f"Synthesis settings or source files changed in {paths.cache}. Use updated config.py paths.")
    else:
        index = {"schema": 1, "status": "building", "data_fingerprint": identity,
                 "settings": settings, "source_schemas": schemas, "sequences": []}
        atomic_json(index_path, index)
    adapter = vendor(device, paths.vendor)
    body = adapter.model(paths.smpl)
    index["status"] = "building"
    atomic_json(index_path, index)
    existing = {row["source"]: row for row in index["sequences"]}
    for relative in selected:
        context.deadline.check()
        previous = existing.get(relative)
        if previous:
            path = paths.cache / previous["cache_file"]
            if path.is_file() and sha256_file(path) == previous["sha256"]:
                context.logger.info("Reuse physical cache %s", relative)
                continue
        source = paths.amass / relative
        sequence = adapter.raw_amass.read_amass_npz(str(source))
        if sequence is None:
            raise RuntimeError(f"Audited upstream loader rejected previously validated file {source}.")
        pose, trans, betas = sequence
        if preset.max_frames_per_sequence is not None:
            pose, trans = pose[:preset.max_frames_per_sequence], trans[:preset.max_frames_per_sequence]
        if len(pose) <= 12:
            raise RuntimeError("The frame cap leaves <=12 frames; increase max_frames_per_sequence in config.py.")
        schema = schemas[relative]
        if abs(schema["effective_fps"] - 60) > 1e-6:
            context.logger.warning("FPS mismatch %s: actual %g, upstream derivative assumes 60 Hz; no resampling applied.",
                                   relative, schema["effective_fps"])
        context.logger.info("SYNTHESIZE %s frames=%d split=%s",relative,len(pose),"train" if schema["subject"] in split["train"] else "validation")
        started = time.monotonic()
        progress_dir = paths.cache / "partials" / identity / fingerprint(relative)[:24]
        signals = synthesize_sequence(adapter, body, (pose, trans, betas), chunk_frames=chunk_frames,
                                      smooth_n=preset.smooth_n, deadline=context.deadline, progress_dir=progress_dir)
        synthesis_seconds = time.monotonic() - started
        cache_file = "sequences/" + fingerprint(relative)[:24] + ".npz"
        target = paths.cache / cache_file
        atomic_write(target, lambda stream: np.savez_compressed(stream, **signals))
        row = {"id": relative.removesuffix("_poses.npz"), "source": relative,
               "subject": schema["subject"], "split": "train" if schema["subject"] in split["train"] else "validation",
               "frames": len(pose), "cache_file": cache_file, "sha256": sha256_file(target)}
        row["synthesis_seconds_this_attempt"] = synthesis_seconds
        row["cache_bytes"] = target.stat().st_size
        existing[relative] = row
        index["sequences"] = [existing[name] for name in selected if name in existing]
        atomic_json(index_path, index)
        shutil.rmtree(progress_dir, ignore_errors=True)
        context.logger.info("Cached %s: %d frames, 24 sites", relative, len(pose))
    for partition in ("train", "validation"):
        if not any(row["split"] == partition for row in index["sequences"]):
            raise RuntimeError(f"No supported cached sequence in {partition}. Review the subject split, frame rates and pilot caps.")
    index["status"] = "complete"
    atomic_json(index_path, index)
    context.logger.info("SYNTHESIS complete: %d sequences; subject split %s", len(index["sequences"]), split)


if __name__=="__main__":raise SystemExit(run_step("1_synthesize",main))

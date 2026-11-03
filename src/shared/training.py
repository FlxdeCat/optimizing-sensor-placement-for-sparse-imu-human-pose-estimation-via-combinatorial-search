"""Fixed-budget configured optimization, restartable checkpoints, sequence scoring, and benchmark.

One ledger identity is shared by every experiment calling evaluate(). A
completed short-budget run can be scored with additional metrics without
retraining. Search always returns frame-weighted validation local angles.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import replace
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import (ARCHITECTURES, PIPELINE_VERSION, SIP_DIP,
                     StopRequested, TrainingBudget, fingerprint, joint_set, CANDIDATE_POOL)
from .data import (AMASSWindowDataset, choose_device, load_cached_sequence,
                         load_index, pack_selected, windows_for_length)
from .storage import RunKey, atomic_json, atomic_write, sha256_file, file_lock
from .metrics import (PoseMetrics, aggregate_sequences, rotation_errors,
                      training_time_seconds)
from .model import build_model, parameter_count


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN LSTM deterministic reproducibility is not guaranteed on every driver.
    # Fixed seeds and saved RNG states are recorded; no claim of bitwise identity.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state() -> dict[str, Any]:
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def runtime_identity(device: torch.device) -> dict[str, Any]:
    return {"device_type": device.type, "torch": str(torch.__version__), "numpy": np.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
            "tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True}


def training_code_fingerprint() -> str:
    """Hash training/scoring helpers and imports, excluding experiment dispatch.

Adding orchestration or benchmark commands cannot change an optimizer
trajectory. Changes to any actual helper remain conservative invalidations.
AST attributes omit line numbers, so moving unchanged code is also harmless.
"""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = {"seed_all", "rng_state", "restore_rng", "synchronize", "runtime_identity",
             "training_code_fingerprint", "pipeline_fingerprint", "training_index", "masked_mse",
             "_oom", "optimizer_step", "_cpu", "_save", "_train", "predict_sequence",
             "score_validation", "evaluate", "ByteLRU", "TrainingSequenceStore",
             "CachedTrainingDataset", "sequence_store", "close_sequence_store"}
    nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) or
             isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names]
    return fingerprint(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False))


def pipeline_fingerprint(device: torch.device) -> str:
    directory = Path(__file__).parent
    return fingerprint({"pipeline_version": PIPELINE_VERSION,
                        "files": {"train_core_ast": training_code_fingerprint(),
                                  **{name: sha256_file(directory / name) for name in
                                     ("model.py", "metrics.py", "data.py", "vendor_adapter.py")}},
                        "runtime": runtime_identity(device)})


def training_index(context) -> dict[str, Any]:
    # One verified immutable cache snapshot per command, rather than reading
    # every multi-gigabyte cache file before every greedy candidate training.
    if not hasattr(context, "_verified_training_index"):
        context._verified_training_index = load_index(context.paths, verify_hashes=True)
    return context._verified_training_index


class ByteLRU:
    """Bound retained array/tensor storage by bytes, never reorder training."""
    def __init__(self, limit_bytes: int):
        self.limit = int(limit_bytes)
        self.bytes = 0
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def put(self, key, value, size: int):
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[1]
        if size > self.limit:
            return
        while self.entries and self.bytes + size > self.limit:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        self.entries[key] = (value, size)
        self.bytes += size

    def statistics(self):
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "retained_bytes": self.bytes, "limit_bytes": self.limit}


class TrainingSequenceStore:
    """Shared, checksummed NPY mirrors of only acc/ori/pose; read-only mmap.

    Physical E1 files remain unchanged. Each mirror is derived once from the
    verified physical sequence. Hashes are checked once per command; arrays are
    opened lazily, avoiding repeated compressed-NPZ decompression.
    """
    def __init__(self, context):
        self.context = context
        self.cache = ByteLRU(context.settings.sequence_cache_mib * 1024 ** 2 // 2)
        self.verified = set()
        self.build_seconds = 0.0
        self.built_sequences = 0
        self._maps = WeakValueDictionary()
        self.closed = False

    def close(self):
        """Close owned root mappings only after all window/metric consumers finish.

        Training/metrics receive copies, so no Torch storage borrows these maps.
        get() results are borrowed and must not be accessed after close(). Weak
        tracking covers evicted/oversized mappings still held by a caller without
        retaining every opened recording in memory.
        """
        if self.closed:
            return
        arrays = list(self._maps.values())
        self.cache.entries.clear()
        self.cache.bytes = 0
        for array in arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()
        self._maps.clear()
        self.verified.clear()
        self.closed = True

    def get(self, row):
        if self.closed:
            raise RuntimeError("Training sequence store is closed; create a new command/dataset context.")
        identity = fingerprint({"schema": 1, "source": row["sha256"], "id": row["id"],
                                "frames": row["frames"], "arrays": "acc-ori-pose-float32"})
        cached = self.cache.get(identity)
        if cached is not None:
            return cached
        directory = self.context.paths.cache / "training_arrays_v1" / identity[:24]
        metadata = directory / "metadata.json"
        expected = {"acc": (row["frames"], 24, 3), "ori": (row["frames"], 24, 3, 3),
                    "pose": (row["frames"], 24, 3, 3)}
        if identity not in self.verified:
            directory.mkdir(parents=True, exist_ok=True)
            with file_lock(directory / "prepare.lock"):
                valid = False
                if metadata.is_file():
                    try:
                        report = json.loads(metadata.read_text())
                        valid = report["identity"] == identity and all(
                            (directory / (name + ".npy")).is_file() and
                            sha256_file(directory / (name + ".npy")) == report["sha256"][name]
                            for name in expected)
                    except (KeyError, ValueError, OSError):
                        valid = False
                if not valid:
                    self.context.deadline.check()
                    started = time.perf_counter()
                    directory.mkdir(parents=True, exist_ok=True)
                    hashes = {}
                    with np.load(self.context.paths.cache / row["cache_file"], allow_pickle=False) as archive:
                        for name, shape in expected.items():
                            if name not in archive:
                                raise ValueError(f"Physical sequence {row['id']} lacks {name}; repair E1.")
                            array = np.asarray(archive[name], dtype=np.float32)
                            if array.shape != shape or not np.isfinite(array).all():
                                raise ValueError(f"Invalid {name} in {row['id']}; repair its physical cache via E1.")
                            target = directory / (name + ".npy")
                            atomic_write(target, lambda stream, array=array: np.save(stream, array, allow_pickle=False))
                            hashes[name] = sha256_file(target)
                    atomic_json(metadata, {"identity": identity, "source_sha256": row["sha256"], "sha256": hashes})
                    self.build_seconds += time.perf_counter() - started
                    self.built_sequences += 1
                    self.context.logger.info("Prepared mmap training arrays for %s", row["id"])
                self.verified.add(identity)
        result = {name: np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
                  for name in expected}
        for array in result.values():
            self._maps[id(array)] = array
        if any(result[name].shape != shape or result[name].dtype != np.float32
               for name, shape in expected.items()):
            raise RuntimeError(f"Training mirror shape/dtype mismatch in {directory}; remove this derived directory and retry.")
        self.cache.put(identity, result, sum(array.nbytes for array in result.values()))
        return result


def sequence_store(context):
    if not hasattr(context, "_training_sequence_store"):
        context._training_sequence_store = TrainingSequenceStore(context)
    return context._training_sequence_store


def close_sequence_store(context):
    """Command/test boundary cleanup; safe and idempotent when no store exists."""
    store = getattr(context, "_training_sequence_store", None)
    if store is not None:
        store.close()
        delattr(context, "_training_sequence_store")


class CachedTrainingDataset(AMASSWindowDataset):
    """Same window selection/order, bounded packed-window and shared mmap caches."""
    def __init__(self, context, index, joints, budget, *, partition, split_seed):
        super().__init__(context.paths, index, joints, budget, partition=partition, split_seed=split_seed)
        self.store = sequence_store(context)
        self.window_cache = ByteLRU(context.settings.sequence_cache_mib * 1024 ** 2 // 2)

    def __getitem__(self, item):
        sequence_index, start, valid_count = self.windows[item]
        key = (sequence_index, start, valid_count)
        cached = self.window_cache.get(key)
        if cached is None:
            data = self.store.get(self.rows[sequence_index])
            # Upstream normalization is framewise. Copy only the window so
            # read-only memmaps cannot become writable Torch tensor storage.
            sl = slice(start, start + valid_count)
            features = pack_selected(data["acc"][sl].copy(), data["ori"][sl].copy(), self.joints)
            rotations = torch.from_numpy(data["pose"][sl].copy()).float()
            cached = (features, rotations)
            self.window_cache.put(key, cached, sum(value.numel() * value.element_size() for value in cached))
        x, y = cached
        padding = self.budget.window_length - valid_count
        if padding:
            x = torch.cat((x, x[-1:].expand(padding, -1)))
            y = torch.cat((y, y[-1:].expand(padding, -1, -1, -1)))
        return {"x": x, "y": y, "valid": torch.arange(self.budget.window_length) < valid_count,
                "sequence": self.rows[sequence_index]["id"], "start": start}


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
               denominator: int | None = None) -> torch.Tensor:
    if (prediction.shape != target.shape or prediction.shape[:2] != valid.shape or
            prediction.shape[-3:] != (24, 3, 3) or valid.dtype != torch.bool):
        raise ValueError("MSE needs matching [B,T,24,3,3] tensors and a [B,T] mask.")
    elements = int(valid.sum()) * 216 if denominator is None else denominator
    if elements < 1:
        raise ValueError("MSE batch has no valid rotation entries.")
    # float32 arithmetic outside autocast keeps matrix-MSE reductions stable.
    squared = (prediction.float() - target.float()).square()
    return (squared * valid[..., None, None, None]).sum() / elements


def _oom(exception: BaseException, device: torch.device) -> bool:
    return device.type == "cuda" and (isinstance(exception, torch.cuda.OutOfMemoryError)
                                     or "out of memory" in str(exception).lower())


def optimizer_step(model, optimizer, scaler, batch: dict[str, Any], device: torch.device,
                   *, amp: bool, microbatch: int, logger) -> tuple[float, int, bool]:
    """Halve physical batch on OOM; accumulate to unchanged effective batch.

Retry from the same RNG state before an optimizer update. Batch=1 OOM requires
an explicit shorter-window protocol, preserving architecture and ledger keys.
GradScaler overflow skips the update; that skip is counted and logged.
"""
    count = len(batch["x"])
    denominator = int(batch["valid"].sum()) * 216
    start_rng = rng_state()
    while True:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        try:
            for start in range(0, count, microbatch):
                x = batch["x"][start:start + microbatch].to(device)
                y = batch["y"][start:start + microbatch].to(device)
                valid = batch["valid"][start:start + microbatch].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp and device.type == "cuda"):
                    prediction = model(x, valid)
                loss = masked_mse(prediction, y, valid, denominator)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite matrix MSE. No update performed; inspect input data and precision.")
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
                del x, y, valid, prediction, loss
        except RuntimeError as exception:
            if not _oom(exception, device):
                raise
            optimizer.zero_grad(set_to_none=True)
            # Remove traceback references to failed GPU activations before retry.
            exception.__traceback__ = None
            x = y = valid = prediction = loss = None
            gc.collect()
            torch.cuda.empty_cache()
            restore_rng(start_rng)
            if microbatch <= 1:
                raise RuntimeError("CUDA OOM at physical batch size 1. Choose a shorter window_length in config.py "
                                   "with updated config.py paths, then rerun E1 and benchmark. "
                                   "The compact BiRNN has hidden size 64; no architecture change is made during OOM recovery.") from exception
            previous = microbatch
            microbatch = max(1, microbatch // 2)
            logger.warning("CUDA OOM: physical batch %d -> %d; effective batch remains %d via gradient accumulation",
                           previous, microbatch, count)
            continue
        # OOM retries cover only forward/backward, before any weight update.
        # An optimizer failure propagates and resumes from the durable step.
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        if skipped:
            logger.warning("AMP gradient overflow: optimizer update skipped, scale=%g", scaler.get_scale())
        return loss_sum, microbatch, skipped


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _save(path: Path, model, optimizer, scaler, state: dict[str, Any]) -> None:
    writes = state.get("checkpoint_writes", 0)
    state["checkpoint_writes"] = writes + 1
    try:
        state.update(model=_cpu(model.state_dict()), optimizer=_cpu(optimizer.state_dict()),
                     scaler=scaler.state_dict(), rng=rng_state())
        atomic_write(path, lambda stream: torch.save(state, stream))
    except BaseException:
        state["checkpoint_writes"] = writes
        raise
    finally:
        for name in ("model", "optimizer", "scaler", "rng"):
            state.pop(name, None)


def _train(context, model, dataset, key: RunKey, budget: TrainingBudget,
           device: torch.device, checkpoint: Path) -> dict[str, Any]:
    optimizer = (torch.optim.SGD(model.parameters(), lr=budget.learning_rate, momentum=budget.momentum, weight_decay=budget.weight_decay, foreach=False)
                 if budget.optimizer == "SGD" else torch.optim.Adam(model.parameters(), lr=budget.learning_rate, weight_decay=budget.weight_decay, foreach=False))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and context.settings.amp)
    state: dict[str, Any] = {"schema": 1, "key": key.id, "identity": key.identity,
                            "epoch": 0, "next_batch": 0, "microbatch": budget.effective_batch_size,
                            "training_seconds": 0.0, "steps": 0, "skipped_updates": 0,
                            "epoch_loss_sum": 0.0, "epoch_batches": 0, "history": [],
                            "runtime": runtime_identity(device), "parameters": parameter_count(model)}
    if checkpoint.exists():
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if loaded.get("schema") != 1 or loaded.get("key") != key.id:
            raise RuntimeError(f"Checkpoint identity mismatch at {checkpoint}. Use a new run root.")
        model.load_state_dict(loaded.pop("model"))
        optimizer.load_state_dict(loaded.pop("optimizer"))
        scaler.load_state_dict(loaded.pop("scaler"))
        restore_rng(loaded.pop("rng"))
        state = loaded
        context.logger.info("Resume %s epoch=%d next_batch=%d", key.id[:12], state["epoch"], state["next_batch"])
    else:
        _save(checkpoint, model, optimizer, scaler, state)
    model.train()
    last_save = time.monotonic()
    step_in_flight = False
    try:
        for epoch in range(state["epoch"], budget.epochs):
            generator = torch.Generator().manual_seed(key.seed)
            for _ in range(epoch + 1):
                order = torch.randperm(len(dataset), generator=generator).tolist()
            first = state["next_batch"] * budget.effective_batch_size
            loader = DataLoader(Subset(dataset, order[first:]), batch_size=budget.effective_batch_size,
                                shuffle=False, num_workers=0, pin_memory=False,
                                generator=torch.Generator().manual_seed(key.seed))
            for batch in loader:
                context.deadline.check()
                step_in_flight = True
                synchronize(device)
                started = time.perf_counter()
                loss, physical, skipped = optimizer_step(model, optimizer, scaler, batch, device,
                                                          amp=context.settings.amp, microbatch=state["microbatch"],
                                                          logger=context.logger)
                synchronize(device)
                state["training_seconds"] += time.perf_counter() - started
                state["microbatch"] = physical
                state["next_batch"] += 1
                state["steps"] += 1
                state["skipped_updates"] += int(skipped)
                state["epoch_loss_sum"] += loss
                state["epoch_batches"] += 1
                step_in_flight = False
                if time.monotonic() - last_save >= context.settings.checkpoint_interval_seconds:
                    _save(checkpoint, model, optimizer, scaler, state)
                    last_save = time.monotonic()
                    context.logger.info("CHECKPOINT %s epoch=%d/%d batch=%d/%d optimizer_s=%.2f physical_batch=%d",key.id[:12],epoch+1,budget.epochs,state["next_batch"],math.ceil(len(dataset)/budget.effective_batch_size),state["training_seconds"],state["microbatch"])
            step_in_flight = True
            state["history"].append({"epoch": epoch + 1, "mean_batch_mse":
                                     state["epoch_loss_sum"] / max(1, state["epoch_batches"]),
                                     "cumulative_training_seconds": state["training_seconds"]})
            state.update(epoch=epoch + 1, next_batch=0, epoch_loss_sum=0.0, epoch_batches=0)
            step_in_flight = False
            _save(checkpoint, model, optimizer, scaler, state)
            last_save = time.monotonic()
            context.logger.info("%s epoch %d/%d mse=%.6g cumulative_train_s=%.2f physical_batch=%d",
                                key.architecture, epoch + 1, budget.epochs, state["history"][-1]["mean_batch_mse"],
                                state["training_seconds"], state["microbatch"])
    except (StopRequested, KeyboardInterrupt):
        if not step_in_flight:
            _save(checkpoint, model, optimizer, scaler, state)
            context.logger.info("Saved completed training work on graceful stop at step %d", state["steps"])
        else:
            context.logger.warning("Interrupted optimizer step or epoch transition not committed; resume from last durable checkpoint")
        raise
    return state


@torch.no_grad()
def predict_sequence(model, features: torch.Tensor, *, window: int, stride: int,
                     device: torch.device, amp: bool, deadline=None,
                     progress_path: Path | None = None, identity: str = "",
                     progress_interval_seconds: float = 60.0) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != 288 or len(features) < 1:
        raise ValueError("Sequence features must be [T,288].")
    model.eval()
    windows = windows_for_length(len(features), window, stride)
    sums = torch.zeros((len(features), 24, 3, 3), dtype=torch.float64)
    counts = torch.zeros(len(features), dtype=torch.int64)
    next_window = 0
    if progress_path is not None and progress_path.exists():
        state = torch.load(progress_path, map_location="cpu", weights_only=True)
        if (state.get("identity") != identity or
                not isinstance(state.get("sums"), torch.Tensor) or tuple(state["sums"].shape) != tuple(sums.shape) or
                not isinstance(state.get("counts"), torch.Tensor) or tuple(state["counts"].shape) != tuple(counts.shape) or
                state["sums"].dtype != torch.float64 or state["counts"].dtype != torch.int64 or
                type(state.get("next_window")) is not int or not 0 <= state["next_window"] <= len(windows)):
            raise RuntimeError("Prediction progress identity changed; choose a new run root.")
        sums, counts, next_window = state["sums"], state["counts"], state["next_window"]
    if not math.isfinite(progress_interval_seconds) or progress_interval_seconds <= 0:
        raise ValueError("Prediction save interval must be finite and positive.")
    last_save = time.monotonic()
    def save_progress():
        if progress_path is not None:
            state = {"identity": identity, "sums": sums, "counts": counts, "next_window": next_window}
            atomic_write(progress_path, lambda stream: torch.save(state, stream))
    window_in_flight = False
    try:
        for index in range(next_window, len(windows)):
            if deadline is not None:
                deadline.check()
            start, valid = windows[index]
            window_in_flight = True
            x = features[start:start + valid].unsqueeze(0).to(device)
            try:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                    prediction = model(x)
                sums[start:start + valid] += prediction[0].double().cpu()
                counts[start:start + valid] += 1
                next_window = index + 1
                window_in_flight = False
            except RuntimeError as exception:
                if _oom(exception, device):
                    raise RuntimeError("Inference OOM at batch 1; use a shorter window with a new protocol/run root.") from exception
                raise
            finally:
                x = prediction = None
            if time.monotonic() - last_save >= progress_interval_seconds:
                save_progress()
                import logging
                logging.getLogger("imuplace").info("PREDICTION progress %d/%d windows (%s)",next_window,len(windows),progress_path)
                last_save = time.monotonic()
    except (StopRequested, KeyboardInterrupt):
        if not window_in_flight:
            save_progress()
        # An interrupted accumulator update must never be saved. Resume its
        # previous durable window snapshot; recent complete windows may replay.
        raise
    if bool((counts == 0).any()):
        raise RuntimeError("Inference window coverage omitted frames.")
    if progress_path is not None:
        progress_path.unlink(missing_ok=True)
    # Average unconstrained matrices over overlapping windows, then project in
    # metrics. The default stride equals window, so no overlap is introduced.
    return (sums / counts[:, None, None, None]).float()


def score_validation(context, model, index: dict[str, Any], key: RunKey, budget: TrainingBudget,
                     device: torch.device, checkpoint: Path, *, full_metrics: bool) -> dict[str, Any]:
    checkpoint_hash = sha256_file(checkpoint)
    directory = context.paths.runs / "validation" / key.id
    directory.mkdir(parents=True, exist_ok=True)
    metric_engine = None
    rows = []
    for sequence in index["sequences"]:
        if sequence["split"] != "validation":
            continue
        context.deadline.check()
        name = fingerprint(sequence["id"])[:24]
        identity = fingerprint({"key": key.id, "sequence": sequence["sha256"], "checkpoint": checkpoint_hash,
                                "full_metrics": full_metrics})
        row_path = directory / (name + (".full.json" if full_metrics else ".local.json"))
        if row_path.exists():
            row = json.loads(row_path.read_text(encoding="utf-8"))
            if row["identity"] != identity:
                raise RuntimeError(f"Validation cache changed at {row_path}. Use a new run root.")
            rows.append(row)
            continue
        data = sequence_store(context).get(sequence)
        prediction_path = directory / (name + ".predictions.pt")
        prediction_identity = fingerprint({"key": key.id, "checkpoint": checkpoint_hash, "sequence": sequence["sha256"]})
        if prediction_path.exists():
            saved = torch.load(prediction_path, map_location="cpu", weights_only=True)
            if saved["identity"] != prediction_identity:
                raise RuntimeError("Prediction cache changed; use a new run root.")
            predictions = saved["predictions"]
        else:
            predictions = predict_sequence(model, pack_selected(data["acc"].copy(), data["ori"].copy(), key.joints),
                                           window=budget.window_length, stride=budget.stride, device=device,
                                           amp=context.settings.amp, deadline=context.deadline,
                                           progress_path=directory / (name + ".inference.partial.pt"), identity=prediction_identity,
                                           progress_interval_seconds=context.settings.prediction_interval_seconds)
            atomic_write(prediction_path, lambda stream: torch.save(
                {"identity": prediction_identity, "predictions": predictions}, stream))
        if full_metrics:
            if metric_engine is None:
                metric_engine = PoseMetrics(context.paths.smpl, fps=context.settings.fps, chunk_frames=context.settings.metric_chunk_frames)
            metrics = metric_engine(predictions, data["pose"].copy(), deadline=context.deadline,
                                    progress_path=directory / (name + ".metrics.partial.pt"), identity=identity,
                                    progress_interval_seconds=context.settings.prediction_interval_seconds)
        else:
            metrics = rotation_errors(predictions, data["pose"].copy())
        row = {"id": sequence["id"], "identity": identity, "frames": len(predictions), **metrics,
               "prediction_file": prediction_path.relative_to(context.paths.runs).as_posix(),
               "prediction_sha256": sha256_file(prediction_path)}
        atomic_json(row_path, row)
        rows.append(row)
    equal = aggregate_sequences(rows)
    weighted = aggregate_sequences(rows, frame_weighted=True)
    result = {"local_rotation_deg": weighted["local_rotation_deg"],
              "sequence_mean": equal, "frame_mean": weighted, "per_sequence": rows,
              "full_metrics": full_metrics}
    atomic_json(directory / ("full_summary.json" if full_metrics else "local_summary.json"), result)
    return result


def evaluate(context, architecture: str, joints: Any, seed: int, *, stage: str = "search",
             full_metrics: bool = False) -> dict[str, Any]:
    """Return a verified completed ledger record; callers never retrain duplicates."""
    sites=joint_set(joints)
    if sites[0]!=0 or not set(sites)<=set(CANDIDATE_POOL):raise ValueError("Training inputs must contain pelvis and belong to the approved 13-site pool.")
    device = choose_device(context.settings.device)
    budget = context.settings.budget(stage)
    index = training_index(context)
    key = RunKey.create(architecture, joint_set(joints), seed, budget.identity,
                        data_fingerprint=index["data_fingerprint"], pipeline_fingerprint=pipeline_fingerprint(device))
    completed = context.ledger.completed(key)
    if completed is not None and (not full_metrics or completed["payload"]["metrics"].get("full_metrics")):
        context.logger.info("Ledger reuse %s %s seed=%d budget=%s", architecture, key.joints, seed, budget.id[:12])
        return completed
    context.deadline.check()
    seed_all(seed)
    model = build_model(architecture, budget.model).to(device)
    checkpoint = context.paths.runs / "checkpoints" / (key.id + ".pt")
    context.ledger.record(key, "started", {"stage": stage, "metrics_upgrade": completed is not None})
    try:
        dataset = CachedTrainingDataset(context, index, key.joints, budget,
                                     partition="train", split_seed=context.settings.split_seed)
        state = _train(context, model, dataset, key, budget, device, checkpoint)
        results = score_validation(context, model, index, key, budget, device, checkpoint, full_metrics=full_metrics)
        results["training_seconds"] = training_time_seconds(state["training_seconds"])
        results["parameters"] = state["parameters"]
        details = {"stage": stage, "train_windows": len(dataset), "history": state["history"],
                   "physical_batch_size": state["microbatch"], "effective_batch_size": budget.effective_batch_size,
                   "skipped_updates": state["skipped_updates"], "runtime": state["runtime"],
                   "validation_partition_used_for_search_and_reporting": True,
                   "independent_synthetic_test_partition": False}
        if results["full_metrics"]:
            for row in results["per_sequence"]:
                if row["rank_deficient_fraction"] > 0:
                    context.logger.warning("%s rank-deficient output fraction=%.6g: nearest rotation can be nonunique",
                                           row["id"], row["rank_deficient_fraction"])
        return context.ledger.record_completed(key, metrics=results, checkpoint=checkpoint, details=details)
    except (StopRequested, KeyboardInterrupt):
        context.ledger.record(key, "interrupted", {"checkpoint": str(checkpoint), "stage": stage})
        raise
    except Exception as exception:
        context.ledger.record(key, "failed", {"error": str(exception), "stage": stage})
        raise
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


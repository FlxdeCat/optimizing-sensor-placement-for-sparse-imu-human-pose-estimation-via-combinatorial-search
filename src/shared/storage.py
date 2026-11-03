"""Atomic output, OS-released locks, and a recoverable JSON-lines evaluation ledger."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, BinaryIO, Callable, Iterator

from config import ARCHITECTURES, canonical_json, fingerprint, joint_set


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_atomic(source: Path, destination: Path) -> None:
    """Retry Windows access/sharing/lock failures without deleting the destination.

    A retry does not bypass persistent filesystem permissions or a read-only
    destination. Total backoff is bounded to 7.15 seconds; other OS errors fail
    immediately. The writer has already closed its own temporary stream.
    """
    delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            if attempt == len(delays):
                raise RuntimeError(
                    f"Cannot atomically replace {destination} after {attempt + 1} attempts. "
                    "The previous destination was not deleted. Close applications reading this file, "
                    "check its read-only attribute and folder modify/delete permissions, or choose "
                    "a writable run/cache directories in src/config.py. Original OS error: " + str(exc)
                ) from exc
            if attempt == 0:
                logging.getLogger("imuplace").warning(
                    "Atomic replacement blocked for %s (Windows error %s); retrying for up to 7.15s.",
                    destination, getattr(exc, "winerror", None))
            time.sleep(delays[attempt])


def atomic_write(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    """Writer receives a binary stream. torch.save(payload, stream) is supported."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_atomic(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError as exc:
            # A scanner may still hold the temporary file. Preserve the main
            # replacement/writer exception and disclose the leftover for cleanup.
            logging.getLogger("imuplace").warning("Temporary file cleanup failed for %s: %s", temp, exc)


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(json.loads(canonical_json(value)), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(payload))


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Non-blocking single-writer lock, automatically released if a process dies.

    The lock file is retained: removing it can create a race between two inodes.
    Locking a run directory for the whole command prevents simultaneous trainings.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    locked = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError(f"Another process holds {path}. Stop it before resuming this run.") from exc
        yield
    finally:
        try:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


@dataclass(frozen=True)
class RunKey:
    architecture: str
    joints: tuple[int, ...]
    seed: int
    budget_json: str
    data_fingerprint: str
    pipeline_fingerprint: str

    @classmethod
    def create(cls, architecture: str, joints: Any, seed: int, budget: dict[str, Any],
               *, data_fingerprint: str, pipeline_fingerprint: str) -> RunKey:
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a non-negative integer.")
        if not budget or not data_fingerprint or not pipeline_fingerprint:
            raise ValueError("A run key requires budget, data, and pipeline fingerprints.")
        return cls(architecture, joint_set(joints), seed, canonical_json(budget),
                   data_fingerprint, pipeline_fingerprint)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture, "joints": list(self.joints), "seed": self.seed,
            "budget": json.loads(self.budget_json), "data_fingerprint": self.data_fingerprint,
            "pipeline_fingerprint": self.pipeline_fingerprint,
        }

    @property
    def id(self) -> str:
        return fingerprint(self.identity)


class Ledger:
    """No incomplete evaluation is considered complete. Checkpoints are verified.

    A torn final row is removed under a lock. Malformed newline-terminated rows
    cause an error rather than discarding historical experiment results.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.records: list[dict[str, Any]] = []
        self.reload()

    def _read(self, *, repair: bool) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        content = self.path.read_bytes()
        if content and not content.endswith(b"\n"):
            cutoff = content.rfind(b"\n") + 1
            if repair:
                with self.path.open("r+b") as stream:
                    stream.truncate(cutoff)
                    stream.flush()
                    os.fsync(stream.fileno())
            content = content[:cutoff]
        records = []
        for number, line in enumerate(content.splitlines(), 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("schema") != 1:
                    raise ValueError("Unknown ledger schema")
                identity = record["identity"]
                if fingerprint(identity) != record["key"]:
                    raise ValueError("Run-key digest does not match identity")
                if record["status"] not in ("started", "interrupted", "failed", "completed"):
                    raise ValueError("Unknown ledger status")
                canonical_json(record)
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"Corrupt ledger row {number} in {self.path}; restore a backup before resuming.") from exc
            records.append(record)
        return records

    def reload(self) -> None:
        with file_lock(self.lock_path):
            self.records = self._read(repair=True)

    def record(self, key: RunKey, status: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if status not in ("started", "interrupted", "failed", "completed"):
            raise ValueError(f"Unknown ledger status: {status}")
        if status == "completed":
            raise ValueError("Use record_completed() so the checkpoint is verified.")
        return self._append(key, status, payload or {})

    def _append(self, key: RunKey, status: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "schema": 1, "key": key.id, "identity": key.identity, "status": status,
            "time_utc": datetime.now(timezone.utc).isoformat(), "payload": payload,
        }
        line = (canonical_json(record) + "\n").encode("utf-8")
        with file_lock(self.lock_path):
            records = self._read(repair=True)
            with self.path.open("ab") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            self.records = [*records, record]
        return record

    def record_completed(self, key: RunKey, *, metrics: dict[str, Any],
                         checkpoint: Path, details: dict[str, Any] | None = None) -> dict[str, Any]:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"Cannot complete evaluation: checkpoint is missing at {checkpoint}.")
        try:
            relative = checkpoint.relative_to(self.path.parent.resolve())
        except ValueError as exc:
            raise ValueError("Checkpoints must be inside the ledger's run directory.") from exc
        payload = {
            "metrics": metrics, "details": details or {},
            "checkpoint": relative.as_posix(), "checkpoint_sha256": sha256_file(checkpoint),
        }
        return self._append(key, "completed", payload)

    def completed(self, key: RunKey) -> dict[str, Any] | None:
        """Return verified completion; refuse lost/changed completed weights.

None means this run has no completed evaluation. A completed ledger row with
an invalid checkpoint must never silently fall back to fresh training.
"""
        for record in reversed(self.records):
            if record["key"] != key.id or record["status"] != "completed":
                continue
            payload = record["payload"]
            checkpoint = (self.path.parent / payload["checkpoint"]).resolve()
            if not checkpoint.is_relative_to(self.path.parent.resolve()):
                raise RuntimeError("Ledger checkpoint path escapes its run directory.")
            if checkpoint.is_file() and sha256_file(checkpoint) == payload["checkpoint_sha256"]:
                return record
            raise RuntimeError(
                f"Completed checkpoint missing or changed at {checkpoint}. Restore its verified "
                "file from backup or start a new protocol in fresh roots; automatic retraining is refused."
            )
        return None

    def completed_records(self) -> list[dict[str, Any]]:
        """Latest completed row per key; use completed(key) before training reuse."""
        result = {}
        for record in self.records:
            if record["status"] == "completed":
                result[record["key"]] = record
        return list(result.values())


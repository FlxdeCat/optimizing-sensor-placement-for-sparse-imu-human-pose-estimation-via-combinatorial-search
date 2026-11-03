"""Script setup, scoped report identities, logging, and graceful cleanup."""
from __future__ import annotations
from dataclasses import dataclass
import csv, io, json, logging, sys
from pathlib import Path
from typing import Any
import config
from config import SETTINGS,Paths,Deadline,StopRequested,canonical_json,fingerprint
from .storage import Ledger,atomic_json,atomic_write,file_lock,sha256_file


@dataclass
class Session:
    settings: Any
    paths: Paths
    deadline: Deadline
    ledger: Ledger
    logger: logging.Logger


def make_session(step,settings=None,paths=None):
    cfg=settings or config.SETTINGS;cfg.validate()
    if paths is None:
        base=Paths(config.PROJECT,config.DATA_ROOT,config.CACHE_ROOT/config.ACTIVE_PRESET,config.RUN_ROOT/config.ACTIVE_PRESET)
        # Config-only model/plot/tolerance changes do not affect this namespace.
        files=sorted(base.amass.rglob("*_poses.npz")) if base.amass.is_dir() else []
        inventory=[(p.relative_to(base.amass).as_posix(),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
        settings_data={"source_root":str(base.amass),"inventory":inventory,"smpl":sha256_file(base.smpl) if base.smpl.is_file() else "missing",
            "pipeline":config.PIPELINE_VERSION,"parents":config.PARENTS,"split_seed":cfg.seed,"train_fraction":cfg.train_subject_fraction,
            "caps":[cfg.max_train_subjects,cfg.max_val_subjects,cfg.max_sequences_per_subject,cfg.max_frames_per_sequence],
            "fps":cfg.fps,"smooth_n":cfg.smooth_n,"fk_chunk":cfg.fk_chunk_frames,"device":cfg.device,
            "code":{name:sha256_file(Path(__file__).with_name(name)) for name in ("data.py","vendor_adapter.py")},
            "synthesis_step":sha256_file(config.PROJECT/"src"/"1_synthesize.py")}
        paths=Paths(base.project,base.data,base.cache/("synthetic_"+fingerprint(settings_data)[:16]),base.runs)
    for folder in (paths.cache,paths.runs,paths.runs/"checkpoints",paths.figures):folder.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger("imuplace");logger.setLevel(logging.INFO);logger.propagate=False
    for handler in list(logger.handlers):handler.close();logger.removeHandler(handler)
    formatter=logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(paths.runs/(step+".log"),encoding="utf-8"),logging.StreamHandler()):
        handler.setFormatter(formatter);logger.addHandler(handler)
    session=Session(cfg,paths,Deadline(cfg.max_minutes),Ledger(paths.ledger),logger)
    logger.info("START %s preset=%s seed=%d device=%s",step,cfg.name,cfg.seed,cfg.device)
    logger.info("DATA %s | CACHE %s | RUN %s",paths.data,paths.cache,paths.runs)
    config_source=config.PROJECT/"src"/"config.py"
    snapshot=paths.runs/"config_snapshots"/(sha256_file(config_source)+".py")
    if not snapshot.exists():atomic_write(snapshot,lambda stream:stream.write(config_source.read_bytes()))
    atomic_json(paths.runs/(step+"_config.json"),{"settings":cfg.identity,"paths":{"data":str(paths.data),"cache":str(paths.cache),"run":str(paths.runs),"amass":str(paths.amass),"smpl":str(paths.smpl),"dip":str(paths.dip),"totalcapture":str(paths.totalcapture),"totalcapture_smpl":str(paths.totalcapture_smpl)},"config_snapshot":snapshot.relative_to(paths.runs).as_posix(),"tolerances_deg":config.GREEDY_TOLERANCES_DEG})
    return session


def run_step(name,operation):
    session=None
    try:
        session=make_session(name)
        with file_lock(session.paths.runs/".script.lock"):
            operation(session)
        session.logger.info("COMPLETE %s",name)
        return 0
    except StopRequested as exc:
        (session.logger.info if session else print)(str(exc));return 0
    except KeyboardInterrupt:
        (session.logger.warning if session else print)("Interrupted. Rerun the same script to resume durable progress.");return 130
    except Exception as exc:
        if session:session.logger.exception("FAILED %s: %s",name,exc)
        else:print(f"FAILED {name}: {exc}",file=sys.stderr)
        return 2
    finally:
        if session:
            if hasattr(session,"_training_sequence_store"):
                from .training import close_sequence_store
                close_sequence_store(session)
            for handler in list(session.logger.handlers):handler.close();session.logger.removeHandler(handler)


def report_path(session,name,settings):
    return session.paths.runs/"reports"/name/(fingerprint(settings)+".json")


def save_report(session,name,report):
    if report.get("identity")!=fingerprint(report["settings"]):raise ValueError("Report identity/settings mismatch.")
    atomic_json(report_path(session,name,report["settings"]),report)
    atomic_json(session.paths.runs/(name+".json"),report)


def load_report(session,name,settings,*,optional=False):
    path=report_path(session,name,settings)
    if not path.is_file():
        if optional:return None
        raise RuntimeError(f"Missing current {name} report. Run the corresponding numbered script with current config.py settings.")
    value=json.loads(path.read_text())
    if value["identity"]!=fingerprint(settings) or canonical_json(value["settings"])!=canonical_json(settings):raise RuntimeError(f"Corrupt report {path}.")
    return value


def write_csv(path,rows):
    path=Path(path);fields=list(dict.fromkeys(k for row in rows for k in row))
    text=io.StringIO(newline="");writer=csv.DictWriter(text,fieldnames=fields)
    if fields:writer.writeheader();writer.writerows([{k:canonical_json(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()} for row in rows])
    payload=text.getvalue().encode("utf-8");atomic_write(path,lambda stream:stream.write(payload))


def artifact(session,relative,digest=None):
    root=session.paths.runs.resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise RuntimeError(f"Missing/unsafe artifact {relative}. Resume its producer script.")
    if digest is not None and sha256_file(path)!=digest:raise RuntimeError(f"Artifact changed: {path}. Restore it from backup.")
    return path

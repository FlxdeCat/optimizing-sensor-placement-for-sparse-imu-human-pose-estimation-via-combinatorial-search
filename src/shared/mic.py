"""Mandatory minepy. No fallback or approximate NumPy estimator exists here."""
from pathlib import Path
import importlib.metadata
import json,time,hashlib
import numpy as np
from config import fingerprint,CANDIDATE_POOL
from .training import training_index,sequence_store
from .storage import atomic_json,atomic_write,sha256_file
from .session import load_report,save_report


def require_minepy(cfg):
    try:
        from minepy import MINE
        obj=MINE(alpha=cfg.mic_alpha,c=cfg.mic_clumps,est=cfg.mic_estimator)
        x=np.arange(32,dtype=float);obj.compute_score(x,x)
        value=float(obj.mic())
        if not np.isfinite(value) or not 0<=value<=1.000001:raise ValueError("Invalid MIC probe result.")
        return MINE,importlib.metadata.version("minepy")
    except Exception as exc:
        raise RuntimeError(f"minepy must import AND calculate successfully. Run install_minepy.ps1 with this virtual environment, then rerun 0_check.py. No fallback is permitted. Original error: {exc}") from exc


def matrix_settings(ctx):
    _,version=require_minepy(ctx.settings)
    import minepy.mine as native
    return {"data":training_index(ctx)["data_fingerprint"],"pool":list(CANDIDATE_POOL),"samples":ctx.settings.mic_samples,"seed":ctx.settings.seed,
            "alpha":ctx.settings.mic_alpha,"clumps":ctx.settings.mic_clumps,"estimator":ctx.settings.mic_estimator,
            "minepy":version,"native_sha256":sha256_file(Path(native.__file__)),"numpy":np.__version__,"channels":"physical acc3 + global orientation9",
            "code":sha256_file(Path(__file__))}


def matrix_report(ctx):
    report=load_report(ctx,"mic",matrix_settings(ctx))
    if report["status"]!="complete":raise RuntimeError("MIC matrix incomplete. Resume 6_greedy_b.py.")
    path=Path(report["matrix_file"])
    if not path.is_file() or sha256_file(path)!=report["sha256"]:raise RuntimeError("Cached MIC matrix missing/changed. Restore it or rerun 6_greedy_b.py to repair.")
    return report


def get_matrix(ctx):
    settings=matrix_settings(ctx);identity=fingerprint(settings)
    previous=load_report(ctx,"mic",settings,optional=True)
    directory=ctx.paths.cache/"mic"/identity[:20];directory.mkdir(parents=True,exist_ok=True)
    target=directory/"matrix.npz"
    if previous and previous["status"]=="complete" and target.is_file() and sha256_file(target)==previous["sha256"]:
        with np.load(target,allow_pickle=False) as d:return d["matrix"].copy(),previous
    sample_path=directory/"samples.npy"
    if sample_path.is_file():samples=np.load(sample_path,allow_pickle=False)
    else:
        rows=[r for r in training_index(ctx)["sequences"] if r["split"]=="train"]
        totals=np.cumsum([r["frames"] for r in rows]);count=min(ctx.settings.mic_samples,int(totals[-1]))
        if count<4:raise ValueError("At least four training frames are required for MIC.")
        offsets=np.sort(np.random.default_rng(ctx.settings.seed).choice(int(totals[-1]),count,replace=False))
        samples=np.empty((count,len(CANDIDATE_POOL),12),dtype=np.float64)
        for i,row in enumerate(rows):
            ctx.deadline.check();low=0 if i==0 else totals[i-1];selected=np.flatnonzero((offsets>=low)&(offsets<totals[i]))
            if not len(selected):continue
            data=sequence_store(ctx).get(row);frames=offsets[selected]-low
            samples[selected]=np.concatenate((data["acc"][frames][:,list(CANDIDATE_POOL)],data["ori"][frames][:,list(CANDIDATE_POOL)].reshape(len(frames),len(CANDIDATE_POOL),9)),axis=-1)
        atomic_write(sample_path,lambda stream:np.save(stream,samples,allow_pickle=False))
    if samples.ndim!=3 or samples.shape[1:]!=(len(CANDIDATE_POOL),12) or not np.isfinite(samples).all():raise ValueError("Invalid MIC samples.")
    progress=directory/"progress.npz";matrix=np.full((len(CANDIDATE_POOL),len(CANDIDATE_POOL)),np.nan);done=np.zeros((len(CANDIDATE_POOL),len(CANDIDATE_POOL)),dtype=bool)
    if progress.exists():
        with np.load(progress,allow_pickle=False) as d:matrix=d["matrix"].copy();done=d["done"].copy()
    if matrix.shape!=(len(CANDIDATE_POOL),len(CANDIDATE_POOL)) or done.shape!=(len(CANDIDATE_POOL),len(CANDIDATE_POOL)) or done.dtype!=np.bool_ or not np.array_equal(done,done.T) or not np.isfinite(matrix[done]).all():raise RuntimeError("Invalid MIC progress; remove only this derived MIC directory and retry.")
    MINE,_=require_minepy(ctx.settings);mine=MINE(alpha=ctx.settings.mic_alpha,c=ctx.settings.mic_clumps,est=ctx.settings.mic_estimator)
    count_done=int(np.triu(done).sum());started=time.perf_counter();memo={}
    stamps=[[hashlib.sha256(samples[:,a,x].tobytes()).digest() for x in range(12)] for a in range(len(CANDIDATE_POOL))]
    timer_path=directory/"timing.json"
    previous_seconds=json.loads(timer_path.read_text())["active_seconds"] if timer_path.exists() else 0.0
    def save():atomic_write(progress,lambda stream:np.savez_compressed(stream,matrix=matrix,done=done))
    try:
        for a in range(len(CANDIDATE_POOL)):
            for b in range(a,len(CANDIDATE_POOL)):
                if done[a,b]:continue
                ctx.deadline.check();values=[]
                for x in range(12):
                    for y in range(12):
                        ctx.deadline.check();v=samples[:,a,x];w=samples[:,b,y]
                        if np.ptp(v)==0 or np.ptp(w)==0:score=0.0
                        else:
                            pair=tuple(sorted((stamps[a][x],stamps[b][y])))
                            if pair not in memo:
                                mine.compute_score(np.ascontiguousarray(v),np.ascontiguousarray(w));memo[pair]=float(mine.mic())
                            score=memo[pair]
                        if not np.isfinite(score) or not 0<=score<=1.000001:raise RuntimeError("minepy returned an invalid value; computation aborted.")
                        values.append(score)
                matrix[a,b]=matrix[b,a]=float(np.mean(values));done[a,b]=done[b,a]=True;count_done+=1
                ctx.logger.info("MIC sensor pair %d/91 sites=(%d,%d) score=%.5f",count_done,CANDIDATE_POOL[a],CANDIDATE_POOL[b],matrix[a,b])
                if count_done%ctx.settings.mic_checkpoint_pairs==0:save()
    finally:
        save()
        atomic_json(timer_path,{"active_seconds":previous_seconds+time.perf_counter()-started})
    atomic_write(target,lambda stream:np.savez_compressed(stream,matrix=matrix,pool=np.array(CANDIDATE_POOL)))
    report={"identity":identity,"settings":settings,"status":"complete","matrix_file":str(target.resolve()),"sha256":sha256_file(target),"active_seconds":previous_seconds+time.perf_counter()-started,"backend":"minepy","samples":len(samples)}
    save_report(ctx,"mic",report);return matrix,report

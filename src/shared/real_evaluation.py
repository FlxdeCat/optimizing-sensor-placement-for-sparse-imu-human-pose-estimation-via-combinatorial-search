"""Frozen-model inference on a shared real cohort. No optimizer is constructed."""
from pathlib import Path
import json,gc,time
import numpy as np
import scipy
import torch
import mapping
from config import fingerprint,joint_set,CANDIDATE_POOL
from .session import artifact,load_report,save_report
from .storage import atomic_json,atomic_write,sha256_file
from .data import choose_device,pack_selected
from .real_data import (RealSequence,UnusableSequence,load_dip_sequence,load_tc_sequence,discover_tc,
                        find_reference,calibration_paths)
from .metrics import PoseMetrics,aggregate_sequences
from .model import build_model
from .training import predict_sequence


def optional_vicon(root,subject,motion):
    matches=[p for p in root.rglob("gt_skel_gbl_pos.txt") if p.parent.name.lower()==motion and p.parent.parent.name.lower()==subject]
    if len(matches)>1:raise RuntimeError(f"Duplicate optional Vicon files for {subject}/{motion}.")
    return matches[0] if matches else None


def cohort_inputs(ctx,dataset):
    rows=[]
    if dataset=="dip":
        for subject in ctx.settings.dip_test_subjects:
            files=sorted((ctx.paths.dip/subject).glob("*.pkl"))
            if not files:raise RuntimeError(f"Missing DIP test files: {ctx.paths.dip/subject}")
            for i,path in enumerate(files):
                if ctx.settings.max_sequences_per_subject is not None and i>=ctx.settings.max_sequences_per_subject:break
                rows.append({"id":subject+"/"+path.stem,"files":[str(path.resolve())]})
    else:
        counts={}
        for subject,motion,path in discover_tc(ctx.paths.totalcapture):
            if subject not in ctx.settings.tc_subjects:continue
            count=counts.get(subject,0)
            if ctx.settings.max_sequences_per_subject is not None and count>=ctx.settings.max_sequences_per_subject:continue
            counts[subject]=count+1
            reference=find_reference(ctx.paths.totalcapture_smpl,subject,motion)
            calibration=calibration_paths(path)
            vicon=optional_vicon(ctx.paths.totalcapture,subject,motion)
            rows.append({"id":subject+"/"+motion,"files":[str(p.resolve()) for p in (path,reference,*calibration)],"vicon":None if vicon is None else str(vicon.resolve())})
    if dataset=="totalcapture":
        seen={r["id"].split("/")[0] for r in rows}
        missing=set(ctx.settings.tc_subjects)-seen
        if missing:raise RuntimeError(f"Missing enabled TotalCapture subjects {sorted(missing)} below {ctx.paths.totalcapture}. Populate them or explicitly change tc_subjects in config.py.")
    for row in rows:
        ctx.deadline.check()
        for file in row["files"]:
            if not Path(file).is_file():raise RuntimeError(f"Missing real input {file}; see README TotalCapture formats/calibration.")
        row["sha256"]=[sha256_file(Path(p)) for p in row["files"]]
        if row.get("vicon"):row["vicon_sha256"]=sha256_file(Path(row["vicon"]))
    return rows


def evaluation_settings(ctx,final,cohort):
    return {"final":final["identity"],"cohort":cohort["identity"],"dataset":cohort["settings"]["dataset"],
            "code":sha256_file(Path(__file__)),
            "step":sha256_file(ctx.paths.project/"src"/"9_evaluation.py")}


def cohort_settings(ctx,dataset):
    from . import real_data
    return {"dataset":dataset,"declared_subjects":list(ctx.settings.dip_test_subjects if dataset=="dip" else ctx.settings.tc_subjects),"excluded_pairs":[] if dataset=="dip" else [list(v) for v in ctx.settings.tc_excluded],"sources":cohort_inputs(ctx,dataset),"mapping":mapping.mapping_fingerprint(),"reader":sha256_file(Path(real_data.__file__)),
            "numpy":np.__version__,"scipy":scipy.__version__,"max_frames":ctx.settings.max_frames_per_sequence,
            "fill_passes":ctx.settings.dip_fill_passes,"trim":ctx.settings.dip_trim_frames,
            "tc_known_pose":ctx.settings.tc_first_reference_pose_calibration,"tc_allow_numeric":ctx.settings.tc_allow_numeric_order}


def prepare_cohort(ctx,dataset):
    mapping.require_approval();settings=cohort_settings(ctx,dataset);inputs=settings["sources"]
    name="cohort_"+dataset
    report=load_report(ctx,name,settings,optional=True) or {"identity":fingerprint(settings),"settings":settings,"status":"running","rows":[]}
    directory=ctx.paths.runs/"real_cohorts"/dataset/report["identity"][:16];directory.mkdir(parents=True,exist_ok=True)
    rows={r["id"]:r for r in report["rows"]}
    for source in inputs:
        ctx.deadline.check()
        if source["id"] in rows:
            old=rows[source["id"]]
            if old["status"]=="ready":artifact(ctx,old["file"],old["sha256"])
            continue
        try:
            if dataset=="dip":sequence=load_dip_sequence(Path(source["files"][0]),source["id"])
            else:
                subject,motion=source["id"].split("/")
                sequence=load_tc_sequence(Path(source["files"][0]),Path(source["files"][1]),None if not source.get("vicon") else Path(source["vicon"]),subject,motion)
        except UnusableSequence as exc:
            row={"id":source["id"],"status":"excluded","reason":str(exc)};ctx.logger.warning("COHORT excluded %s: %s",source["id"],exc)
        else:
            count=len(sequence.pose) if ctx.settings.max_frames_per_sequence is None else min(len(sequence.pose),ctx.settings.max_frames_per_sequence)
            fields={key:getattr(sequence,key)[:count] for key in ("acc","ori","pose","trans")}
            if sequence.vicon is not None:fields["vicon"]=sequence.vicon[:count]
            target=directory/(fingerprint(source["id"])[:24]+".npz")
            atomic_write(target,lambda stream:np.savez_compressed(stream,**fields))
            row={"id":source["id"],"status":"ready","frames":count,"file":target.relative_to(ctx.paths.runs).as_posix(),"sha256":sha256_file(target),"diagnostics":sequence.diagnostics,"vicon_names":list(sequence.vicon_names)}
            ctx.logger.info("COHORT ready %s %s frames=%d",dataset,source["id"],count)
        rows[source["id"]]=row;report["rows"]=[rows[k] for k in sorted(rows)];save_report(ctx,name,report)
    ready=[r for r in rows.values() if r["status"]=="ready"]
    if not ready:raise RuntimeError(f"No usable {dataset} sequences. Inspect the cohort report.")
    report.update(status="complete",sequence_count=len(ready),frames=sum(r["frames"] for r in ready))
    save_report(ctx,name,report);return report


def load_prepared(ctx,dataset,row):
    with np.load(artifact(ctx,row["file"],row["sha256"]),allow_pickle=False) as arrays:data={k:arrays[k].copy() for k in arrays.files}
    return RealSequence(row["id"],dataset,data["acc"],data["ori"],data["pose"],data["trans"],row["diagnostics"],data.get("vicon"),tuple(row["vicon_names"]))


def realization(dataset,joints,proxy=False):
    if proxy:raise ValueError("Head-for-neck substitution is not part of the reviewed 13-site protocol.")
    chosen=joint_set(joints)
    if not (len(chosen)==6 or chosen==CANDIDATE_POOL):
        return {"realizable":False,"reason":"Expected six selected sites or exactly the all-13 reference."}
    report=mapping.realizability(dataset,chosen)
    return {**report,"sensors":report["sensor_indices"],"head_for_neck_proxy":False}


def features(sequence,joints,proxy=False):
    report=realization(sequence.dataset,joints,proxy)
    if not report["realizable"]:raise ValueError(f"Non-realizable configuration {report}")
    chosen=joint_set(joints);n=len(sequence.pose)
    acc=np.zeros((n,24,3),np.float32);ori=np.zeros((n,24,3,3),np.float32)
    acc[:,list(chosen)]=sequence.acc[:,report["sensors"]];ori[:,list(chosen)]=sequence.ori[:,report["sensors"]]
    return pack_selected(acc,ori,chosen)


def evaluate_real(ctx,record,cohort,proxy=False):
    from . import metrics,data,model
    joints=record["identity"]["joints"];dataset=cohort["settings"]["dataset"]
    checkpoint=artifact(ctx,record["payload"]["checkpoint"],record["payload"]["checkpoint_sha256"])
    settings={"training":record["key"],"checkpoint":record["payload"]["checkpoint_sha256"],"cohort":cohort["identity"],
        "mapping":mapping.mapping_fingerprint(),"proxy":proxy,"amp":ctx.settings.amp,"fps":ctx.settings.fps,"smpl":sha256_file(ctx.paths.smpl),
        "code":{Path(module.__file__).name:sha256_file(Path(module.__file__)) for module in (metrics,data,model)},"real_code":sha256_file(Path(__file__))}
    identity=fingerprint(settings);directory=ctx.paths.runs/"real_predictions"/identity[:24];directory.mkdir(parents=True,exist_ok=True)
    summary=directory/"summary.json";device=choose_device(ctx.settings.device)
    if summary.exists():
        saved=json.loads(summary.read_text())
        for row in saved["per_sequence"]:artifact(ctx,row["prediction_file"],row["prediction_sha256"])
        return saved
    network=engine=None;rows=[];started=time.perf_counter()
    try:
        for source in cohort["rows"]:
            if source["status"]!="ready":continue
            ctx.deadline.check();tag=fingerprint(source["id"])[:24];row_identity=fingerprint([identity,source["sha256"]]);row_path=directory/(tag+".json")
            if row_path.exists():
                row=json.loads(row_path.read_text())
                if row["identity"]!=row_identity:raise RuntimeError("Real sequence identity mismatch.")
                artifact(ctx,row["prediction_file"],row["prediction_sha256"]);rows.append(row);continue
            sequence=load_prepared(ctx,dataset,source);prediction_path=directory/(tag+".pt")
            if prediction_path.exists():
                saved=torch.load(prediction_path,map_location="cpu",weights_only=True)
                if saved["identity"]!=row_identity:raise RuntimeError("Real prediction identity mismatch.")
                prediction=saved["predictions"]
            else:
                if network is None:
                    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
                    if saved["key"]!=record["key"] or saved["epoch"]!=record["identity"]["budget"]["epochs"]:raise RuntimeError("Expected a completed final checkpoint.")
                    network=build_model("birnn",ctx.settings.model);network.load_state_dict(saved["model"]);del saved;network=network.to(device).eval()
                budget=record["identity"]["budget"]
                ctx.logger.info("REAL %s %s joints=%s sensors=%s fixed pelvis anchor, /30, zero mask, proxy=%s",dataset,sequence.id,joints,realization(dataset,joints,proxy)["sensors"],proxy)
                prediction=predict_sequence(network,features(sequence,joints,proxy),window=budget["window_length"],stride=budget["stride"],device=device,amp=ctx.settings.amp,
                    deadline=ctx.deadline,progress_path=directory/(tag+".inference.partial.pt"),identity=row_identity,progress_interval_seconds=ctx.settings.prediction_interval_seconds)
                atomic_write(prediction_path,lambda stream:torch.save({"identity":row_identity,"predictions":prediction},stream))
            if engine is None:engine=PoseMetrics(ctx.paths.smpl,chunk_frames=ctx.settings.metric_chunk_frames,fps=ctx.settings.fps)
            values=engine(prediction,sequence.pose,deadline=ctx.deadline,progress_path=directory/(tag+".metrics.partial.pt"),identity=row_identity,progress_interval_seconds=ctx.settings.prediction_interval_seconds)
            row={"id":sequence.id,"identity":row_identity,**values,"prediction_file":prediction_path.relative_to(ctx.paths.runs).as_posix(),"prediction_sha256":sha256_file(prediction_path),"prepared_file":source["file"],"prepared_sha256":source["sha256"]}
            atomic_json(row_path,row);rows.append(row);ctx.logger.info("REAL score %s %s MPJPE=%.4f cm local=%.4f deg",dataset,sequence.id,values["mpjpe_cm"],values["local_rotation_deg"])
    finally:
        del network,engine;gc.collect()
        if device.type=="cuda":torch.cuda.empty_cache()
    report={"identity":identity,"settings":settings,"status":"complete","per_sequence":rows,"sequence_mean":aggregate_sequences(rows),"frame_mean":aggregate_sequences(rows,frame_weighted=True),"wall_seconds_this_invocation":time.perf_counter()-started}
    atomic_json(summary,report);return report

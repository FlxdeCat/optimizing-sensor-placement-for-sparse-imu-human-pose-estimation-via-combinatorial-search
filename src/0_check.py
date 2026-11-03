"""Inspect environment, raw formats and enabled datasets; no model training."""
import importlib.metadata,json,platform,subprocess,sys,time
from pathlib import Path
from config import fingerprint
import torch
import mapping
from shared.session import run_step
from shared.storage import atomic_json,sha256_file
from shared.mic import require_minepy
from shared.vendor_adapter import verify_vendor,load_smpl_payload
from shared.data import discover_amass,inspect_amass,amass_subject
from shared.real_data import inspect_dip,discover_tc,find_reference,read_tc_raw,read_tc_calibration,calibration_paths,canonical_tc_order,read_tc_reference,read_vicon


def main(ctx):
    report={"python":sys.version,"platform":platform.platform(),"torch":str(torch.__version__),"compiled_cuda":torch.version.cuda,
            "packages":{name:importlib.metadata.version(name) for name in ("torch","numpy","scipy","matplotlib","opencv-python-headless","rich","minepy","Cython","setuptools","wheel")},
            "cuda_available":torch.cuda.is_available(),"amass":[],"dip":[],"totalcapture":[],"problems":[]}
    target=ctx.paths.runs/"environment_and_data_check.json"
    from shared import real_data,data
    cache_id=fingerprint({"step":sha256_file(Path(__file__)),"reader":sha256_file(Path(real_data.__file__)),"amass_reader":sha256_file(Path(data.__file__)),"mapping":mapping.mapping_fingerprint(),"fill":ctx.settings.dip_fill_passes,"trim":ctx.settings.dip_trim_frames})
    progress=ctx.paths.runs/"data_check_progress"/(cache_id+".json")
    cache=json.loads(progress.read_text()) if progress.exists() else {}
    last_save=time.monotonic()
    def checked(path,operation):
        nonlocal last_save
        key=str(path.resolve());stat=path.stat();stamp=[stat.st_size,stat.st_mtime_ns]
        saved=cache.get(key)
        if saved and saved["stamp"]==stamp:return saved["summary"]
        value=operation();cache[key]={"stamp":stamp,"summary":value}
        if time.monotonic()-last_save>=ctx.settings.checkpoint_interval_seconds:
            atomic_json(progress,cache);last_save=time.monotonic()
        return value
    try:
        if sys.version_info[:2]!=(3,11):raise RuntimeError("Use the documented Python 3.11 environment.")
        try:
            result=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15)
            report["nvidia_smi"]={"returncode":result.returncode,"output":result.stdout.strip(),"error":result.stderr.strip()}
            ctx.logger.info("GPU nvidia-smi: %s",result.stdout.strip() or result.stderr.strip())
        except (OSError,subprocess.TimeoutExpired) as exc:report["nvidia_smi"]={"error":str(exc)}
        if ctx.settings.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable. Install the documented PyTorch CUDA build and inspect nvidia-smi.")
        if torch.cuda.is_available():report["gpu"]={"name":torch.cuda.get_device_name(),"memory_mib":torch.cuda.get_device_properties(0).total_memory//(1024**2)}
        MINE,version=require_minepy(ctx.settings);report["minepy"]=version
        ctx.logger.info("minepy %s imported and completed a numerical probe; no approximation backend.",version)
        verify_vendor(ctx.paths.vendor);body=load_smpl_payload(ctx.paths.smpl)
        report["smpl"]={k:list(v.shape) for k,v in body.items()}
        files=discover_amass(ctx.paths.amass)
        for i,file in enumerate(files,1):
            ctx.deadline.check()
            try:report["amass"].append({"file":str(file),**checked(file,lambda file=file:inspect_amass(file))})
            except Exception as exc:report["problems"].append(f"{file}: {exc}")
            if i%50==0:ctx.logger.info("CHECK CMU %d/%d files",i,len(files));atomic_json(target,report)
        if len({r["subject"] for r in report["amass"] if r["supported"]})<2:report["problems"].append("Need at least two supported CMU capture groups.")
        if ctx.settings.dip_enabled:
            for subject in ctx.settings.dip_test_subjects:
                files=sorted((ctx.paths.dip/subject).glob("*.pkl"))
                if not files:report["problems"].append(f"Missing raw DIP files in {ctx.paths.dip/subject}")
                for file in files:
                    ctx.deadline.check();ctx.logger.info("CHECK DIP %s",file)
                    try:report["dip"].append(checked(file,lambda file=file:inspect_dip(file)))
                    except Exception as exc:report["problems"].append(f"{file}: {exc}")
        if ctx.settings.totalcapture_enabled:
            for subject,motion,file in discover_tc(ctx.paths.totalcapture):
                if subject not in ctx.settings.tc_subjects:continue
                ctx.deadline.check();ctx.logger.info("CHECK TC %s/%s",subject,motion)
                try:
                    raw=read_tc_raw(file);canonical_tc_order(raw["labels"],allow_numeric=ctx.settings.tc_allow_numeric_order)
                    for calibration in calibration_paths(file):read_tc_calibration(calibration,raw["labels"])
                    reference=find_reference(ctx.paths.totalcapture_smpl,subject,motion);_,meta=read_tc_reference(reference)
                    from shared.real_evaluation import optional_vicon
                    vicon=optional_vicon(ctx.paths.totalcapture,subject,motion)
                    diagnostic=None
                    if vicon is not None:
                        names,positions=read_vicon(vicon);diagnostic={"file":str(vicon),"joint_names":list(names),"positions_shape":list(positions.shape)}
                    report["totalcapture"].append({"id":subject+"/"+motion,"labels":raw["labels"],"reference":meta,"optional_vicon":diagnostic})
                except Exception as exc:report["problems"].append(f"{file}: {exc}")
        else:ctx.logger.info("TotalCapture disabled: no access/files required for current execution.")
        if ctx.settings.totalcapture_enabled:
            seen={r["id"].split("/")[0] for r in report["totalcapture"]}
            for subject in sorted(set(ctx.settings.tc_subjects)-seen):report["problems"].append(f"Missing usable TotalCapture subject {subject}; populate data or change enabled subjects explicitly.")
        report["mapping"]=mapping.baseline_report()
        from shared.mapping_review import write_review
        write_review(ctx)
        if report["problems"]:raise RuntimeError(f"Data check found {len(report['problems'])} problems; inspect {target}.")
        report["status"]="complete"
    finally:
        atomic_json(progress,cache);atomic_json(target,report)
    ctx.logger.info("Environment/data report saved: %s",target)


if __name__=="__main__":raise SystemExit(run_step("0_check",main))

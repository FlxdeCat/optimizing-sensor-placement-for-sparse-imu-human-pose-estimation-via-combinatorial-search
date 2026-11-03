"""Descriptive summaries only: no hypothesis tests or confidence intervals."""
from pathlib import Path
import json
import numpy as np
from config import fingerprint,JOINT_NAMES,tolerance_values,tolerance_label,main_tolerance
from .selection import final_results,random_settings,greedy_settings,completed_record
from .session import load_report,save_report,write_csv,artifact
from .storage import sha256_file
from .real_evaluation import evaluation_settings,cohort_settings

METRICS=("local_rotation_deg","global_rotation_deg","sip_deg","mpjpe_cm","mesh_cm","jitter_km_s3")


def sources(ctx):
    final,records=final_results(ctx)
    evaluations=[]
    for dataset in ctx.settings.datasets:
        path=ctx.paths.runs/("evaluation_"+dataset+".json")
        if not path.exists():
            ctx.logger.warning("ANALYSIS %s not evaluated; synthetic summaries remain available.",dataset);continue
        real=json.loads(path.read_text());cohort=real["cohort"]
        # Reading analysis must not construct a cohort or infer predictions.
        if cohort["settings"]!=cohort_settings(ctx,dataset):raise RuntimeError("Real inputs, mapping or preprocessing changed; rerun 9_evaluation.py.")
        expected=evaluation_settings(ctx,final,cohort)
        if real["settings"]!=expected or real["status"]!="complete":raise RuntimeError(f"Current {dataset} evaluation missing/incomplete. Run 9_evaluation.py or disable that dataset temporarily.")
        for row in real["rows"]:
            for seq in row.get("per_sequence",[]):artifact(ctx,seq["prediction_file"],seq["prediction_sha256"])
        evaluations.append(real)
    random=load_report(ctx,"random_search",random_settings(ctx))
    greedy=[load_report(ctx,"greedy_a",greedy_settings(ctx,"a"))]
    greedy += [load_report(ctx,tolerance_label(t),greedy_settings(ctx,"b",t)) for t in tolerance_values()]
    settings={"final":final["identity"],"evaluations":[r["identity"] for r in evaluations],"random":random["identity"],
              "greedy":[g["identity"] for g in greedy],"main_tolerance_deg":main_tolerance(),"code":sha256_file(Path(__file__))}
    return final,records,evaluations,random,greedy,settings


def percentile(value,distribution):
    array=np.asarray(distribution,float)
    # Lower error is better. Midrank of strict-better and tied random draws.
    return float(100*(np.count_nonzero(array<value)+.5*np.count_nonzero(array==value))/len(array))


def build(ctx):
    final,records,reals,random,greedies,settings=sources(ctx)
    rows=[]
    for item in final["rows"]:
        record=records[item["run_key"]];metric=record["payload"]["metrics"]
        rows.append({**item,"dataset":"synthetic_validation","condition":"original","status":"complete",
                     "sequence_mean":metric["sequence_mean"],"frame_mean":metric["frame_mean"],"per_sequence":metric["per_sequence"],
                     "training_seconds":metric["training_seconds"],"parameters":metric["parameters"]})
    for real in reals:
        for item in real["rows"]:
            metric=records[item["run_key"]]["payload"]["metrics"]
            rows.append({**item,"training_seconds":metric["training_seconds"],"parameters":metric["parameters"]})
    groups={}
    for row in rows:groups.setdefault((row["dataset"],row["label"],row["condition"]),[]).append(row)
    summaries=[];sequences=[]
    for (dataset,label,condition),items in sorted(groups.items()):
        item=items[0];available=[r for r in items if r["status"]=="complete"]
        out={"dataset":dataset,"label":label,"condition":condition,"family":item["family"],"joints":item["joints"],
             "joint_names":[JOINT_NAMES[j] for j in item["joints"]],"seeds":[r["seed"] for r in available],"seed_count":len(available),"is_main_greedy_b":label==tolerance_label(main_tolerance()),
             "status":"complete" if available else "non_realizable","parameters":item["parameters"],
             "qualification":"Validation used for selection; not an independent synthetic test." if dataset=="synthetic_validation" else "Frozen synthetic-only model; user-approved segment mappings; DIP Spine3 uses a left-chest proxy."}
        if not available:out["reason"]=item["realizability"]
        for metric in METRICS:
            values=[r["sequence_mean"][metric] for r in available if r["sequence_mean"].get(metric) is not None]
            out[metric]=float(np.mean(values)) if values else None
            out[metric+"_seed_sd"]=float(np.std(values,ddof=1)) if len(values)>1 else None
        out["training_seconds"]=float(np.mean([r["training_seconds"] for r in available])) if available else None
        summaries.append(out)
        for row in available:
            for seq in row["per_sequence"]:
                sequences.append({"dataset":dataset,"label":label,"condition":condition,"seed":row["seed"],"run_key":row["run_key"],"joints":row["joints"],**seq})
    for dataset in sorted({r["dataset"] for r in summaries}):
        pool=[r for r in summaries if r["dataset"]==dataset and r["family"]=="random" and r["condition"]=="original" and r["status"]=="complete"]
        for row in (r for r in summaries if r["dataset"]==dataset):
            for metric in METRICS:
                row[metric+"_random_error_percentile"]=(percentile(row[metric],[p[metric] for p in pool]) if len(pool)==ctx.settings.random_draws and row[metric] is not None and all(p[metric] is not None for p in pool) else None)
    from .swaps import current_reports
    refinements=current_reports(ctx)
    a=greedies[0];overlaps=[];steps=[]
    for g in greedies:
        label="greedy_a" if g["settings"]["variant"]=="a" else tolerance_label(g["settings"]["tau_deg"])
        selected=set(g["selected"]);base=set(a["selected"])
        overlaps.append({"configuration":label,"compared_with":"greedy_a","shared_joints":sorted(selected&base),"jaccard":len(selected&base)/len(selected|base)})
        for step in g["steps"]:
            steps.append({"label":label,"tau_deg":g["settings"]["tau_deg"],**step})
    tie_statistics=[]
    for g in greedies[1:]:
        ties=[s["tie_breaker_triggered"] for s in g["steps"]];changed=[s["tie_breaker_changed_selection"] for s in g["steps"]]
        tie_statistics.append({"label":tolerance_label(g["tau_deg"]),"tau_deg":g["tau_deg"],"steps":len(ties),"trigger_rate":sum(ties)/len(ties),"changed_selection_rate":sum(changed)/len(changed)})
    swap_summary=[];swap_candidates=[]
    for label,refined in refinements:
        initial=refined["initial"];result=refined["current"]
        swap_summary.append({"label":label,"original_joints":initial["joints"],"refined_joints":result["joints"],"initial_error_deg":initial["local_rotation_deg"],"refined_error_deg":result["local_rotation_deg"],"accepted_swaps":refined["accepted_swaps"],"sweeps":len(refined["rounds"]),"stop_reason":refined["stop_reason"],"local_optimum_certified":refined["local_optimum_certified"]})
        selected=set(result["joints"]);base=set(a["selected"])
        overlaps.append({"configuration":label,"compared_with":"greedy_a","shared_joints":sorted(selected&base),"jaccard":len(selected&base)/len(selected|base)})
        for step in refined["rounds"]:
            swap_candidates += [{"label":label,"round":step["round"],"accepted_round":step["accepted"],**r} for r in step["candidates"]]
    search_keys={r["run_key"] for r in random["rows"]}
    search_keys.update(r["run_key"] for r in swap_candidates)
    for g in greedies:
        search_keys.add(g["pelvis_only"]["run_key"])
        search_keys.update(r["run_key"] for s in g["steps"] for r in s["candidates"])
    training=[]
    for key in sorted(search_keys|set(records)):
        record=records.get(key) or completed_record(ctx,key);metrics=record["payload"]["metrics"]
        training.append({"run_key":key,"used_in_search":key in search_keys,"used_in_final":key in records,"seed":record["identity"]["seed"],"joints":record["identity"]["joints"],"epochs":record["identity"]["budget"]["epochs"],"parameters":metrics["parameters"],"optimizer_seconds":metrics["training_seconds"]})
    identity=fingerprint(settings);directory=ctx.paths.runs/"tables"/identity[:16];directory.mkdir(parents=True,exist_ok=True)
    for name,table in (("results",summaries),("per_sequence",sequences),("greedy_steps",steps),("overlap",overlaps),("tie_statistics",tie_statistics),("training_runs",training),("swap_summary",swap_summary),("swap_candidates",swap_candidates)):
        write_csv(directory/(name+".csv"),table)
    report={"identity":identity,"settings":settings,"status":"complete","rows":summaries,"per_sequence":sequences,"greedy_steps":steps,"overlap":overlaps,
            "tie_statistics":tie_statistics,"swap_summary":swap_summary,"training_runs":training,"table_directory":directory.relative_to(ctx.paths.runs).as_posix(),"aggregation":"Equal-sequence mean per seed, then equal mean over enabled seeds; sample SD only with >=2 seeds.",
            "percentile_definition":f"100 * (number of strictly lower random errors + 0.5 * number of equal random errors)/{ctx.settings.random_draws}; lower is better.",
            "unique_final_models":len(records),"unique_final_training_seconds":sum(r["payload"]["metrics"]["training_seconds"] for r in records.values())}
    save_report(ctx,"analysis",report);ctx.logger.info("ANALYSIS wrote %d configuration rows and %d sequence rows to %s",len(summaries),len(sequences),directory)
    return report


def current(ctx):
    *_,settings=sources(ctx)
    return load_report(ctx,"analysis",settings)

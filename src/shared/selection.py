"""Small shared definitions for the readable random/A/B step scripts."""
from pathlib import Path
import json
import numpy as np
from config import BASELINES,fingerprint,joint_set,tolerance_values,tolerance_label,CANDIDATE_POOL
from .data import choose_device
from .training import evaluate,training_index,pipeline_fingerprint
from .session import load_report,save_report,artifact
from .storage import sha256_file,RunKey


def signature(ctx,stage="search"):
    return {"data":training_index(ctx)["data_fingerprint"],"pipeline":pipeline_fingerprint(choose_device(ctx.settings.device)),
            "budget":ctx.settings.budget(stage).identity,"seed":ctx.settings.seed,"pelvis_required":True,"candidate_pool":list(CANDIDATE_POOL)}


def draw_settings(ctx):
    return {"seed":ctx.settings.seed,"count":ctx.settings.random_draws,"pool":list(CANDIDATE_POOL),"fixed":[0],"k":6,"numpy":np.__version__}


def draw_layouts(seed,count):
    rng=np.random.default_rng(seed)
    return [[0,*sorted(int(v) for v in rng.choice(np.array(CANDIDATE_POOL[1:]),5,replace=False))] for _ in range(count)]


def draws(ctx):
    settings=draw_settings(ctx)
    report=load_report(ctx,"random_draws",settings,optional=True)
    if report:return report
    report={"identity":fingerprint(settings),"settings":settings,"draws":draw_layouts(ctx.settings.seed,ctx.settings.random_draws),"status":"complete"}
    save_report(ctx,"random_draws",report);return report


def random_settings(ctx):return {"training":signature(ctx),"draw_identity":draws(ctx)["identity"],"script":sha256_file(ctx.paths.project/"src"/"4_randoms.py")}


def score(ctx,joints):
    chosen=joint_set(joints)
    if chosen[0]!=0 or not set(chosen)<=set(CANDIDATE_POOL):raise ValueError("Pelvis is mandatory and sites must belong to the reviewed pool.")
    record=evaluate(ctx,"birnn",chosen,ctx.settings.seed,stage="search",full_metrics=False)
    return {"joints":list(chosen),"run_key":record["key"],"checkpoint_sha256":record["payload"]["checkpoint_sha256"],
            "local_rotation_deg":record["payload"]["metrics"]["local_rotation_deg"],
            "training_seconds":record["payload"]["metrics"]["training_seconds"]}


def greedy_settings(ctx,variant,tau=None):
    settings={"training":signature(ctx),"variant":variant,"tau_deg":tau,
              "script":sha256_file(ctx.paths.project/"src"/("5_greedy_a.py" if variant=="a" else "6_greedy_b.py")),
              "selection_code":sha256_file(Path(__file__))}
    if variant=="b":
        a=load_report(ctx,"greedy_a",greedy_settings(ctx,"a"))
        if a["status"]!="complete":raise RuntimeError("Complete 5_greedy_a.py, inspect its gap review, then set the tolerance.")
        from .mic import matrix_report
        settings.update(calibration_path=a["identity"],mic_identity=matrix_report(ctx)["identity"])
    return settings


def choose_candidate(candidates,selected,tau=0.0,matrix=None):
    if not candidates:raise ValueError("No candidates.")
    if any(r["added_joint"] not in CANDIDATE_POOL or r["added_joint"] in selected or not np.isfinite(r["local_rotation_deg"]) for r in candidates):raise ValueError("Invalid candidate/site or non-finite score.")
    if not np.isfinite(tau) or tau<0:raise ValueError("Tolerance must be finite and nonnegative.")
    if matrix is not None and (matrix.shape!=(len(CANDIDATE_POOL),len(CANDIDATE_POOL)) or not np.isfinite(matrix).all()):raise ValueError("MIC matrix must be finite and ordered by CANDIDATE_POOL.")
    best=min(candidates,key=lambda r:(r["local_rotation_deg"],r["added_joint"]))
    tied=[r for r in candidates if r["local_rotation_deg"]<=best["local_rotation_deg"]+tau]
    components=[]
    if matrix is not None and len(tied)>1:
        for row in tied:
            j=row["added_joint"];remaining=[v for v in CANDIDATE_POOL if v not in selected and v!=j]
            lookup={site:i for i,site in enumerate(CANDIDATE_POOL)}
            relevance=float(matrix[lookup[j],[lookup[v] for v in remaining]].mean()) if remaining else 0.0
            redundancy=float(matrix[lookup[j],[lookup[v] for v in selected]].mean()) if selected else 0.0
            components.append({"joint":j,"relevance":relevance,"redundancy":redundancy,"score":relevance-redundancy,"error":row["local_rotation_deg"]})
        winner=min(components,key=lambda r:(-r["score"],r["error"],r["joint"]))["joint"]
    else:winner=best["added_joint"]
    chosen=next(r for r in candidates if r["added_joint"]==winner)
    return chosen,{"minimum_error_deg":best["local_rotation_deg"],"minimum_error_joint":best["added_joint"],
        "tie_set":[r["added_joint"] for r in tied],"tie_breaker_triggered":matrix is not None and len(tied)>1,
        "tie_breaker_changed_selection":winner!=best["added_joint"],"tie_scores":components}


def catalog(ctx):
    random=load_report(ctx,"random_search",random_settings(ctx))
    a=load_report(ctx,"greedy_a",greedy_settings(ctx,"a"))
    if random["status"]!="complete" or a["status"]!="complete":raise RuntimeError("Complete random search and Greedy A before 8_retrain.py.")
    items=[{"label":f"random_{i:02d}","family":"random","joints":sites} for i,sites in enumerate(draws(ctx)["draws"])]
    items += [{"label":name,"family":"baseline","joints":list(sites)} for name,sites in BASELINES.items()]
    items += [{"label":"upper_13","family":"upper","joints":list(CANDIDATE_POOL)},
              {"label":"greedy_a","family":"selected","joints":a["selected"]}]
    source_ids=[random["identity"],a["identity"]]
    for tau in tolerance_values():
        label=tolerance_label(tau);b=load_report(ctx,label,greedy_settings(ctx,"b",tau))
        if b["status"]!="complete":raise RuntimeError(f"Complete 6_greedy_b.py for tau={tau} before retraining.")
        items.append({"label":label,"family":"selected","joints":b["selected"],"tau_deg":tau});source_ids.append(b["identity"])
    from .swaps import current_reports
    for label,refined in current_reports(ctx):
        items.append({"label":label,"family":"refined","joints":refined["current"]["joints"]})
        source_ids.append(refined["identity"])
    return items,{"training":signature(ctx,"final"),"seeds":list(ctx.settings.final_seeds),"selections":source_ids,"items":items}


def completed_record(ctx,key_id):
    record=next((r for r in reversed(ctx.ledger.records) if r["key"]==key_id and r["status"]=="completed"),None)
    if record is None:raise RuntimeError(f"Missing completed training {key_id}. Resume the producer step.")
    identity=record["identity"]
    key=RunKey.create(identity["architecture"],identity["joints"],identity["seed"],identity["budget"],data_fingerprint=identity["data_fingerprint"],pipeline_fingerprint=identity["pipeline_fingerprint"])
    return ctx.ledger.completed(key)


def final_results(ctx):
    items,settings=catalog(ctx)
    report=load_report(ctx,"final_retraining",settings)
    if report["status"]!="complete":raise RuntimeError("Complete 8_retrain.py first.")
    expected={(i["label"],seed) for i in items for seed in ctx.settings.final_seeds}
    if {(r["label"],r["seed"]) for r in report["rows"]}!=expected:raise RuntimeError("Final result catalog is incomplete.")
    checked={}
    for row in report["rows"]:
        if row["run_key"] not in checked:checked[row["run_key"]]=completed_record(ctx,row["run_key"])
        record=checked[row["run_key"]]
        if record["identity"]["joints"]!=row["joints"] or record["identity"]["seed"]!=row["seed"]:raise RuntimeError("Final report/ledger mismatch.")
    return report,checked

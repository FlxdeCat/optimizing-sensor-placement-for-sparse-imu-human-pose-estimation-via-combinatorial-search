"""Measure real epochs/scoring on cached CMU, then project this configured protocol."""
from dataclasses import replace
from pathlib import Path
import json,time,gc
import torch
import config
from config import fingerprint,tolerance_values,CANDIDATE_POOL
from shared.session import run_step,Session,save_report,load_report
from shared.storage import Ledger,RunKey,sha256_file
from shared.data import choose_device
from shared.model import build_model
from shared.training import (training_index,sequence_store,close_sequence_store,seed_all,
    CachedTrainingDataset,_train,score_validation,pipeline_fingerprint,synchronize)
from shared.mic import get_matrix


def main(ctx):
    index=training_index(ctx);device=choose_device(ctx.settings.device)
    try:tolerances=tolerance_values();assumed=False
    except ValueError:
        if config.GREEDY_TOLERANCES_DEG is not None:raise
        tolerances=(None,);assumed=True
    settings={"data":index["data_fingerprint"],"pipeline":pipeline_fingerprint(device),"settings":ctx.settings.identity,
              "planned_tolerances":tolerances,"script":sha256_file(Path(__file__))}
    prior=load_report(ctx,"benchmark",settings,optional=True)
    if prior and prior["status"]=="complete":
        ctx.logger.info("Reusing completed benchmark. Report: %s",ctx.paths.runs/"benchmark.json");return
    report=prior or {"identity":fingerprint(settings),"settings":settings,"status":"running","measurements":[]}
    root=ctx.paths.runs/"benchmarks"/report["identity"][:16];root.mkdir(parents=True,exist_ok=True)
    for row in index["sequences"]:ctx.deadline.check();sequence_store(ctx).get(row)
    report["derived_array_preparation_seconds_this_invocation"]=sequence_store(ctx).build_seconds
    measurements={(r["batch_size"],r["stage"]):r for r in report["measurements"]}
    for batch in ctx.settings.benchmark_batch_sizes:
        for stage in ("search","final"):
            if (batch,stage) in measurements:continue
            child_paths=replace(ctx.paths,runs=root/(stage+f"_batch{batch}"));child_paths.runs.mkdir(parents=True,exist_ok=True)
            child=Session(replace(ctx.settings,batch_size=batch),child_paths,ctx.deadline,Ledger(child_paths.ledger),ctx.logger)
            child._verified_training_index=index
            budget=replace(child.settings.budget(stage),epochs=ctx.settings.benchmark_epochs)
            key=RunKey.create("birnn",(0,4,5,15,18,19),ctx.settings.seed,budget.identity,data_fingerprint=index["data_fingerprint"],pipeline_fingerprint=pipeline_fingerprint(device))
            checkpoint=child.paths.runs/(key.id+".pt")
            # Completed measurements resume. A partially measured epoch is replayed
            # from initialization so its wall duration is not understated.
            checkpoint.unlink(missing_ok=True)
            import shutil
            shutil.rmtree(child.paths.runs/"validation"/key.id,ignore_errors=False) if (child.paths.runs/"validation"/key.id).exists() else None
            network=None
            try:
                for row in index["sequences"]:ctx.deadline.check();sequence_store(child).get(row)
                dataset=CachedTrainingDataset(child,index,key.joints,budget,partition="train",split_seed=ctx.settings.seed)
                seed_all(ctx.settings.seed);network=build_model("birnn",budget.model).to(device)
                if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
                synchronize(device);started=time.perf_counter();state=_train(child,network,dataset,key,budget,device,checkpoint);synchronize(device)
                train_wall=time.perf_counter()-started;started=time.perf_counter()
                score_validation(child,network,index,key,budget,device,checkpoint,full_metrics=stage=="final")
                score_wall=time.perf_counter()-started
                row={"batch_size":batch,"stage":stage,"windows":len(dataset),"epochs_measured":budget.epochs,
                     "epoch_wall_seconds":train_wall/budget.epochs,"optimizer_seconds_per_epoch":state["training_seconds"]/budget.epochs,
                     "scoring_wall_seconds":score_wall,"physical_batch":state["microbatch"],
                     "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated(device)/(1024**2) if device.type=="cuda" else None}
                measurements[batch,stage]=row;report["measurements"]=list(measurements.values());save_report(ctx,"benchmark",report)
                ctx.logger.info("BENCHMARK %s batch=%d windows=%d epoch wall=%.3fs score wall=%.3fs",stage,batch,len(dataset),row["epoch_wall_seconds"],score_wall)
            finally:
                del network;close_sequence_store(child);gc.collect()
                if device.type=="cuda":torch.cuda.empty_cache()
    _,mic=get_matrix(ctx);report["mic_active_seconds_once"]=mic["active_seconds"]
    projections={}
    for batch in ctx.settings.benchmark_batch_sizes:
        search=measurements[batch,"search"];final=measurements[batch,"final"]
        costs={"search":search["epoch_wall_seconds"]*ctx.settings.search_epochs+search["scoring_wall_seconds"],
               "final":final["epoch_wall_seconds"]*ctx.settings.final_epochs+final["scoring_wall_seconds"]}
        paths=len(tolerances);final_labels=ctx.settings.random_draws+4+2*paths;seeds=len(ctx.settings.final_seeds)
        forward_slots=sum(len(CANDIDATE_POOL)-k for k in range(1,6))
        swap_neighbors=5*(len(CANDIDATE_POOL)-6)
        swap_slots=swap_neighbors*ctx.settings.benchmark_swap_sweeps_per_path*(1+paths)
        counts={"0_check":(0,0),"1_synthesize":(0,0),"2_upperbound":(0,1),"3_baseline":(0,1),
                "4_randoms":(ctx.settings.random_draws,0),"5_greedy_a":(forward_slots+1,0),"6_greedy_b":(forward_slots*paths,0),"7_swap":(swap_slots,0),
                "8_retrain":(0,final_labels*seeds-2),"9_evaluation":(0,0),"10_analyze":(0,0),"11_visualizations":(0,0)}
        values={name:{"search_training_slots_before_dedup":s,"additional_final_slots_before_dedup":f,"projected_known_hours":(s*costs["search"]+f*costs["final"])/3600} for name,(s,f) in counts.items()}
        values["7_swap"]["status"]="Scenario only: configured sweeps per A/B path; actual stopping sweep count is unknown."
        values["7_swap"]["scenario_sweeps_per_path"]=ctx.settings.benchmark_swap_sweeps_per_path
        values["6_greedy_b"]["projected_known_hours"]+=mic["active_seconds"]/3600
        values["1_synthesize"]["projected_known_hours"]=report["derived_array_preparation_seconds_this_invocation"]/3600
        values["1_synthesize"]["status"]="Derived training-array preparation only; physical synthesis unmeasured."
        projections[str(batch)]={"steps":values,"total_measured_components_hours":sum(v["projected_known_hours"] for v in values.values()),"cost_seconds_per_run":costs}
    report.update(status="complete",projections=projections,
        planned_greedy_b_paths=len(tolerances),b_path_count_assumed=assumed,
        interpretation="Projection uses the CURRENT configured search/final epochs, not a proposed budget. Before cache deduplication; stage2/3 final runs are excluded from additional step8 count. Pelvis-only adds one search training. Identical subsets, including shared greedy prefixes, reuse the ledger.",
        limitations=["Representative six-sensor layout; 13-sensor model has the same input width/parameters but convergence/timing may vary.",
            "Real inference, initial synthesis, checks, figure rendering and table exports are unmeasured, not zero runtime.",
            "Disk-cache warmth, GPU thermal throttling, background work and sequence lengths affect timing. One epoch gives limited precision.",
            "MIC active time is added once, even if already paid. Derived NPY preparation is recorded separately.",
            "If tolerance is unset, one Greedy B path is assumed. Rebenchmark after entering the tolerance list.",
            "Swap cost is a configurable sweep-count SCENARIO, not a guaranteed total. No finite sweep cap is assumed by default; cache reuse can substantially reduce fresh trainings."])
    save_report(ctx,"benchmark",report)
    for batch,value in projections.items():ctx.logger.info("PROJECTED batch=%s measured protocol components %.2f hours; target %.1f hours; unmeasured components excluded",batch,value["total_measured_components_hours"],ctx.settings.benchmark_target_hours)


if __name__=="__main__":raise SystemExit(run_step("benchmark",main))

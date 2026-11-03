"""Error-only forward selection. Pelvis is fixed; review gaps before running B."""
import numpy as np
from config import JOINT_NAMES,fingerprint,CANDIDATE_POOL
from shared.session import run_step,load_report,save_report,write_csv
from shared.selection import greedy_settings,score,choose_candidate


def main(ctx):
    settings=greedy_settings(ctx,"a")
    report=load_report(ctx,"greedy_a",settings,optional=True) or {"identity":fingerprint(settings),"settings":settings,"status":"running","steps":[],"selected":[0]}
    # A pelvis-only score anchors the k=1 curve; the five additions cost 50 slots.
    if "pelvis_only" not in report:
        ctx.logger.info("GREEDY A scoring the fixed pelvis-only reference")
        report["pelvis_only"]=score(ctx,[0]);save_report(ctx,"greedy_a",report)
    while len(report["selected"])<6:
        ctx.deadline.check();selected=report["selected"];k=len(selected)+1
        existing=report.get("current_step",{});candidates={r["added_joint"]:r for r in existing.get("candidates",[]) if existing.get("before")==selected}
        for joint in CANDIDATE_POOL[1:]:
            if joint in selected:continue
            ctx.deadline.check()
            ctx.logger.info("GREEDY A k=%d add=%d %s (%d/%d candidates)",k,joint,JOINT_NAMES[joint],len(candidates)+int(joint not in candidates),len(CANDIDATE_POOL)-len(selected))
            candidates[joint]={"added_joint":joint,**score(ctx,sorted([*selected,joint]))}
            report["current_step"]={"before":selected,"candidates":list(candidates.values())};save_report(ctx,"greedy_a",report)
        chosen,details=choose_candidate(list(candidates.values()),selected)
        report["steps"].append({"k":k,"before":selected,"selected":chosen["joints"],"chosen_joint":chosen["added_joint"],"chosen_error_deg":chosen["local_rotation_deg"],"candidates":list(candidates.values()),**details})
        report["selected"]=chosen["joints"];report.pop("current_step",None);save_report(ctx,"greedy_a",report)
        ctx.logger.info("GREEDY A chose %s error=%.5f deg",[JOINT_NAMES[j] for j in report["selected"]],chosen["local_rotation_deg"])
    report["status"]="complete";save_report(ctx,"greedy_a",report)
    review=[];all_candidates=[]
    for step in report["steps"]:
        values=sorted(r["local_rotation_deg"] for r in step["candidates"]);minimum=values[0]
        row={"k":step["k"],"candidate_count":len(values),"minimum_deg":minimum,"second_best_gap_deg":values[1]-minimum,"candidate_range_deg":values[-1]-minimum}
        for tau in ctx.settings.threshold_review_degrees:row[f"tie_count_at_{tau:g}_deg"]=sum(v<=minimum+tau for v in values)
        review.append(row)
        all_candidates += [{"k":step["k"],"added_joint":r["added_joint"],"joint_name":JOINT_NAMES[r["added_joint"]],"error_deg":r["local_rotation_deg"],"gap_deg":r["local_rotation_deg"]-minimum,"run_key":r["run_key"]} for r in step["candidates"]]
    write_csv(ctx.paths.runs/"greedy_a_gap_review.csv",review)
    write_csv(ctx.paths.runs/"greedy_a_candidates.csv",all_candidates)
    from shared.plotting import gap_review_plot,save_figure
    save_figure(gap_review_plot(report,ctx.settings.threshold_review_degrees),ctx.paths.figures/"greedy_a_tolerance_review.png",ctx.settings.dpi)
    ctx.logger.info("REVIEW greedy_a_gap_review.csv and figures/greedy_a_tolerance_review.png, then set GREEDY_TOLERANCES_DEG in config.py. No tolerance has been chosen automatically.")


if __name__=="__main__":raise SystemExit(run_step("5_greedy_a",main))

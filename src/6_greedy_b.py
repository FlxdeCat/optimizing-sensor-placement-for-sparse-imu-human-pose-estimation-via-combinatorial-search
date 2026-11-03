"""Dependence-assisted forward selection AFTER manual review of Greedy A gaps."""
from config import JOINT_NAMES,fingerprint,CANDIDATE_POOL,tolerance_values,tolerance_label
from shared.session import run_step,load_report,save_report
from shared.selection import greedy_settings,score,choose_candidate
from shared.mic import get_matrix


def main(ctx):
    tolerances=tolerance_values()  # fail before expensive work if no manual input
    a=load_report(ctx,"greedy_a",greedy_settings(ctx,"a"))
    if a["status"]!="complete":raise RuntimeError("Complete 5_greedy_a.py and inspect its gap review first.")
    matrix,_=get_matrix(ctx)
    for tau in tolerances:
        name=tolerance_label(tau);settings=greedy_settings(ctx,"b",tau)
        report=load_report(ctx,name,settings,optional=True) or {"identity":fingerprint(settings),"settings":settings,"status":"running","tau_deg":tau,"steps":[],"selected":[0],"pelvis_only":a["pelvis_only"]}
        ctx.logger.info("GREEDY B tolerance=%.6g deg. Empirical admissible-error band, not a statistical equivalence test.",tau)
        while len(report["selected"])<6:
            ctx.deadline.check();selected=report["selected"];k=len(selected)+1
            existing=report.get("current_step",{});candidates={r["added_joint"]:r for r in existing.get("candidates",[]) if existing.get("before")==selected}
            for joint in CANDIDATE_POOL[1:]:
                if joint in selected:continue
                ctx.deadline.check()
                ctx.logger.info("GREEDY B tau=%g k=%d add=%d %s",tau,k,joint,JOINT_NAMES[joint])
                candidates[joint]={"added_joint":joint,**score(ctx,sorted([*selected,joint]))}
                report["current_step"]={"before":selected,"candidates":list(candidates.values())};save_report(ctx,name,report)
            chosen,details=choose_candidate(list(candidates.values()),selected,tau=tau,matrix=matrix)
            report["steps"].append({"k":k,"before":selected,"selected":chosen["joints"],"chosen_joint":chosen["added_joint"],"chosen_error_deg":chosen["local_rotation_deg"],"candidates":list(candidates.values()),**details})
            report["selected"]=chosen["joints"];report.pop("current_step",None);save_report(ctx,name,report)
            ctx.logger.info("GREEDY B tau=%g k=%d chosen=%s error=%.5f tie=%s triggered=%s changed=%s",tau,k,[JOINT_NAMES[j] for j in report["selected"]],chosen["local_rotation_deg"],details["tie_set"],details["tie_breaker_triggered"],details["tie_breaker_changed_selection"])
        report["status"]="complete";save_report(ctx,name,report)


if __name__=="__main__":raise SystemExit(run_step("6_greedy_b",main))

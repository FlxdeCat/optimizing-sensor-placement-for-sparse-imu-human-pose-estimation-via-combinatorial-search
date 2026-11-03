"""Final-budget SIP/DIP layout; early reconstruction diagnostics."""
from config import BASELINES,fingerprint
from shared.session import run_step,save_report
from shared.selection import signature
from shared.training import evaluate


def main(ctx):
    settings={"training":signature(ctx,"final"),"baselines":BASELINES}
    rows=[]
    for label,joints in BASELINES.items():
        ctx.logger.info("BASELINE %s sites=%s",label,joints)
        record=evaluate(ctx,"birnn",joints,ctx.settings.seed,stage="final",full_metrics=True)
        rows.append({"label":label,"joints":list(joints),"run_key":record["key"]})
        from shared.plotting import early_diagnostics
        early_diagnostics(ctx,label,record)
    save_report(ctx,"baselines",{"identity":fingerprint(settings),"settings":settings,"status":"complete","rows":rows})
    ctx.logger.info("Inspect figures/baseline_diagnostics before committing to the full search. No automatic model tuning is performed.")


if __name__=="__main__":raise SystemExit(run_step("3_baseline",main))

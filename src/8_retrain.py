"""Final retraining from initialization; identical subset/seed/budget is reused."""
from config import fingerprint
from shared.session import run_step,load_report,save_report
from shared.selection import catalog
from shared.training import evaluate


def main(ctx):
    items,settings=catalog(ctx)
    report=load_report(ctx,"final_retraining",settings,optional=True) or {"identity":fingerprint(settings),"settings":settings,"status":"running","rows":[]}
    rows={(r["label"],r["seed"]):r for r in report["rows"]}
    for item in items:
        for seed in ctx.settings.final_seeds:
            ctx.deadline.check()
            ctx.logger.info("FINAL %s seed=%d sites=%s",item["label"],seed,item["joints"])
            record=evaluate(ctx,"birnn",item["joints"],seed,stage="final",full_metrics=True)
            rows[item["label"],seed]={**item,"seed":seed,"run_key":record["key"]}
            report["rows"]=[rows[k] for k in sorted(rows)];save_report(ctx,"final_retraining",report)
    report.update(status="complete",unique_models=len({r["run_key"] for r in rows.values()}))
    save_report(ctx,"final_retraining",report)


if __name__=="__main__":raise SystemExit(run_step("8_retrain",main))

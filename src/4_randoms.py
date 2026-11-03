"""Train/score 30 uniformly drawn pelvis-plus-five layouts at the search budget."""
from config import fingerprint
from shared.session import run_step,load_report,save_report
from shared.selection import draws,random_settings,score


def main(ctx):
    manifest=draws(ctx);settings=random_settings(ctx)
    report=load_report(ctx,"random_search",settings,optional=True) or {"identity":fingerprint(settings),"settings":settings,"status":"running","rows":[]}
    rows={r["draw_index"]:r for r in report["rows"]}
    for index,joints in enumerate(manifest["draws"]):
        ctx.deadline.check()
        ctx.logger.info("RANDOM %d/%d seed=%d sites=%s",index+1,len(manifest["draws"]),ctx.settings.seed,joints)
        rows[index]={"draw_index":index,**score(ctx,joints)}
        report["rows"]=[rows[k] for k in sorted(rows)];save_report(ctx,"random_search",report)
    report["status"]="complete";save_report(ctx,"random_search",report)


if __name__=="__main__":raise SystemExit(run_step("4_randoms",main))

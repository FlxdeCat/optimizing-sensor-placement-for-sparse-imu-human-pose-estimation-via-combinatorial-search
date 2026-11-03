"""Five-epoch (configured final budget) 13-sensor reference, seed 1234."""
from config import fingerprint,CANDIDATE_POOL
from shared.session import run_step,save_report
from shared.selection import signature
from shared.training import evaluate


def main(ctx):
    settings={"training":signature(ctx,"final"),"joints":list(CANDIDATE_POOL)}
    record=evaluate(ctx,"birnn",CANDIDATE_POOL,ctx.settings.seed,stage="final",full_metrics=True)
    save_report(ctx,"upperbound",{"identity":fingerprint(settings),"settings":settings,"status":"complete","run_key":record["key"]})


if __name__=="__main__":raise SystemExit(run_step("2_upperbound",main))

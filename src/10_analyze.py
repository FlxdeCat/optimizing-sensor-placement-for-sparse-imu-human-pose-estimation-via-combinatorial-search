"""Export descriptive metric tables, random ranks, overlap and tie summaries."""
from shared.session import run_step
from shared.analysis import build


def main(ctx):build(ctx)


if __name__=="__main__":raise SystemExit(run_step("10_analyze",main))

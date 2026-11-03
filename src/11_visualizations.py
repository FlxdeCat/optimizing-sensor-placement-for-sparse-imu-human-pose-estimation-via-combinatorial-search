"""Generate PNG summaries and actual best/worst/random SMPL comparisons."""
from shared.session import run_step,save_report,write_csv
from shared.analysis import current
from shared.plotting import FigureSet,summary_figures,pose_figures


def main(ctx):
    analysis=current(ctx);writer=FigureSet(ctx,analysis)
    summary_figures(ctx,analysis,writer);pose_figures(ctx,analysis,writer)
    write_csv(writer.directory/"pose_examples.csv",writer.report["pose_examples"])
    writer.report["status"]="complete";save_report(ctx,"visualizations",writer.report)
    ctx.logger.info("FIGURES complete: %s",writer.directory)


if __name__=="__main__":raise SystemExit(run_step("11_visualizations",main))

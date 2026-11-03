"""Frozen real evaluation of the unified final catalog, including all-13 and swaps."""
from config import fingerprint
from shared.session import run_step,save_report
from shared.selection import final_results
from shared.real_evaluation import prepare_cohort,realization,evaluate_real,evaluation_settings


def main(ctx):
    final,records=final_results(ctx)
    for dataset in ctx.settings.datasets:
        cohort=prepare_cohort(ctx,dataset)
        settings=evaluation_settings(ctx,final,cohort)
        report={'identity':fingerprint(settings),'settings':settings,'status':'running',
                'dataset':dataset,'cohort':cohort,'rows':[]}
        memo={}
        for item in final['rows']:
            ctx.deadline.check();possible=realization(dataset,item['joints'])
            row={**item,'source_label':item['label'],'dataset':dataset,
                 'condition':'original','realizability':possible}
            if not possible['realizable']:
                row['status']='non_realizable'
            else:
                key=item['run_key']
                if key not in memo:memo[key]=evaluate_real(ctx,records[key],cohort)
                row.update(status='complete',**{k:memo[key][k] for k in ('per_sequence','sequence_mean','frame_mean')})
                row['evaluation_identity']=memo[key]['identity']
            report['rows'].append(row);save_report(ctx,'evaluation_'+dataset,report)
            ctx.logger.info('EVALUATION %s %s seed=%d status=%s',dataset,row['label'],item['seed'],row['status'])
        report['status']='complete';save_report(ctx,'evaluation_'+dataset,report)
    if not ctx.settings.datasets:ctx.logger.info('Both real datasets disabled in config.py; synthetic work is available.')


if __name__=='__main__':raise SystemExit(run_step('9_evaluation',main))

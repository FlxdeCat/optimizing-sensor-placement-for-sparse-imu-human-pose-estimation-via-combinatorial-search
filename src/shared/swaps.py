"""Best-improvement single-swap refinement; same training objective as forward search."""
from pathlib import Path
import math
from config import CANDIDATE_POOL,JOINT_NAMES,joint_set,fingerprint,tolerance_values,tolerance_label
from .session import load_report,save_report,write_csv
from .storage import sha256_file


def source_reports(ctx):
    from .selection import greedy_settings
    sources=[('greedy_a',greedy_settings(ctx,'a'))]
    sources += [(tolerance_label(t),greedy_settings(ctx,'b',t)) for t in tolerance_values()]
    result=[]
    for name,settings in sources:
        report=load_report(ctx,name,settings)
        if report['status']!='complete':raise RuntimeError(f'Complete forward search {name} before 7_swap.py.')
        result.append((name,report))
    return result


def refinement_settings(ctx,name,source):
    from .selection import signature
    return {'training':signature(ctx),'source_name':name,'source':source['identity'],
        'pool':list(CANDIDATE_POOL),'fixed':[0],'method':'best_improvement_error_only',
        'minimum_improvement_deg':ctx.settings.swap_min_improvement_deg,
        'max_accepted':ctx.settings.swap_max_accepted,
        'code':sha256_file(Path(__file__)),'step':sha256_file(ctx.paths.project/'src'/'7_swap.py')}


def current_reports(ctx):
    result=[]
    for name,source in source_reports(ctx):
        report=load_report(ctx,name+'_swap',refinement_settings(ctx,name,source))
        if report['status']!='complete':raise RuntimeError('Complete 7_swap.py before final retraining.')
        result.append((name+'_swap',report))
    return result


def neighbors(selected):
    sites=joint_set(selected,size=6)
    if sites[0]!=0 or not set(sites)<=set(CANDIDATE_POOL):raise ValueError('Swap source must contain pelvis and six reviewed sites.')
    return [(removed,added,sorted([j for j in sites if j!=removed]+[added]))
        for removed in sites if removed!=0 for added in CANDIDATE_POOL if added not in sites]


def best_neighbor(rows):
    if not rows or any(not math.isfinite(r['local_rotation_deg']) for r in rows):raise ValueError('Expected finite candidate errors.')
    return min(rows,key=lambda r:(r['local_rotation_deg'],tuple(r['joints'])))


def improve(current,candidate,threshold):
    return current-candidate>threshold


def run(ctx):
    from .selection import score
    for name,source in source_reports(ctx):
        settings=refinement_settings(ctx,name,source);label=name+'_swap'
        report=load_report(ctx,label,settings,optional=True)
        if report is None:
            initial=score(ctx,source['selected'])
            if not math.isfinite(initial['local_rotation_deg']):raise ValueError('Non-finite initial swap error.')
            report={'identity':fingerprint(settings),'settings':settings,'status':'running','source':name,
                'initial':initial,'current':initial,'rounds':[],'accepted_swaps':0,'local_optimum_certified':False}
            save_report(ctx,label,report)
        while report['status']!='complete':
            ctx.deadline.check()
            cap=ctx.settings.swap_max_accepted
            if cap is not None and report['accepted_swaps']>=cap:
                report.update(status='complete',stop_reason='accepted_swap_cap',local_optimum_certified=False)
                save_report(ctx,label,report);break
            sites=report['current']['joints']
            pending=report.get('pending',{'before':sites,'candidates':[]})
            if pending['before']!=sites:raise RuntimeError('Pending swap sweep disagrees with current subset.')
            rows={(r['removed_joint'],r['added_joint']):r for r in pending['candidates']}
            proposals=neighbors(sites)
            for removed,added,chosen in proposals:
                if (removed,added) in rows:continue
                ctx.deadline.check()
                ctx.logger.info('SWAP %s sweep=%d candidate=%d/%d remove=%d:%s add=%d:%s',name,
                    len(report['rounds'])+1,len(rows)+1,len(proposals),removed,JOINT_NAMES[removed],added,JOINT_NAMES[added])
                row={'removed_joint':removed,'added_joint':added,**score(ctx,chosen)}
                if not math.isfinite(row['local_rotation_deg']):raise ValueError('Non-finite swap score; no decision made.')
                rows[removed,added]=row
                report['pending']={'before':sites,'candidates':list(rows.values())};save_report(ctx,label,report)
            best=best_neighbor(list(rows.values()));before=report['current']
            accepted=improve(before['local_rotation_deg'],best['local_rotation_deg'],ctx.settings.swap_min_improvement_deg)
            report['rounds'].append({'round':len(report['rounds'])+1,'before':before,'best_neighbor':best,
                'accepted':accepted,'improvement_deg':before['local_rotation_deg']-best['local_rotation_deg'],
                'candidates':list(rows.values())})
            report.pop('pending',None)
            if accepted:
                report['current']=best;report['accepted_swaps']+=1
                ctx.logger.info('SWAP ACCEPT %s %.8f -> %.8f degrees',name,before['local_rotation_deg'],best['local_rotation_deg'])
            else:
                exact=ctx.settings.swap_min_improvement_deg==0
                report.update(status='complete',stop_reason='no_improving_neighbor' if exact else 'no_improvement_above_threshold',local_optimum_certified=exact)
                ctx.logger.info('SWAP STOP %s checked %d neighbors; one-swap local certificate=%s',name,len(proposals),exact)
            save_report(ctx,label,report)
        write_csv(ctx.paths.runs/(label+'_candidates.csv'),[
            {'round':r['round'],**c,'delta_from_current_deg':c['local_rotation_deg']-r['before']['local_rotation_deg']}
            for r in report['rounds'] for c in r['candidates']])
        ctx.logger.info('SWAP RESULT %s sites=%s accepted=%d error=%.6f degrees',label,report['current']['joints'],report['accepted_swaps'],report['current']['local_rotation_deg'])

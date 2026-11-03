"""Approved segment mapping tables; no fabricated physical sensor coordinates."""
import mapping
from config import CANDIDATE_POOL,JOINT_NAMES,fingerprint
from .session import write_csv,save_report


def rows():
    mapping.validate_mapping();result=[]
    for j in CANDIDATE_POOL:
        row={'smpl_index':j,'smpl_joint':JOINT_NAMES[j],'decision':'retained by user','notes':''}
        for dataset,names in [('dip',mapping.DIP_SENSOR_NAMES),('totalcapture',mapping.TC_SENSOR_NAMES)]:
            e=mapping.mapping_table(dataset)[j]
            row.update({dataset+'_index':e.sensor,dataset+'_name':names[e.sensor],dataset+'_confidence':e.confidence,dataset+'_reason':e.reason})
        result.append(row)
    return result


def write_review(ctx):
    values=rows();settings={'mapping':mapping.mapping_fingerprint()}
    report={'identity':fingerprint(settings),'settings':settings,'status':'complete','rows':values,
        'qualification':'User-approved segment correspondences, not measured mounting coordinates. DIP Spine3 uses an asymmetric left-chest proxy.'}
    save_report(ctx,'sensor_mapping',report);write_csv(ctx.paths.runs/'sensor_mapping.csv',values)
    inventory=[]
    for dataset,names in [('dip',mapping.DIP_SENSOR_NAMES),('totalcapture',mapping.TC_SENSOR_NAMES)]:
        table=mapping.mapping_table(dataset)
        for i,name in enumerate(names):
            inventory.append({'dataset':dataset,'sensor_index':i,'sensor_name':name,
                'smpl_sites':[j for j in CANDIDATE_POOL if table[j].sensor==i]})
    write_csv(ctx.paths.runs/'physical_sensor_inventory.csv',inventory)
    ctx.logger.info('MAPPING 13 distinct streams per dataset; DIP Spine3 -> left chest is an approved proxy. Tables: %s',ctx.paths.runs/'sensor_mapping.csv')

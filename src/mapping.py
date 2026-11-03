"""User-approved 13-site segment correspondences; DIP chest remains a proxy.

Approval permits execution; it does not establish exact mounting coordinates or
independent calibration validity. Joint rotations describe the distal segment.
"""
from dataclasses import dataclass,asdict
from config import BASELINES,JOINT_NAMES,CANDIDATE_POOL,SETTINGS,fingerprint,joint_set
APPROVED = SETTINGS.mapping_acknowledged
TC_TEST_SUBJECTS = SETTINGS.tc_subjects
TC_EXCLUDED = SETTINGS.tc_excluded
DIP_TEST_SUBJECTS = SETTINGS.dip_test_subjects

@dataclass(frozen=True)
class MappingEntry:
    sensor: int | None
    reason: str
    confidence: str

DIP_SENSOR_NAMES = (
    "head", "spine2", "belly", "lchest", "rchest", "lshoulder", "rshoulder",
    "lelbow", "relbow", "lhip", "rhip", "lknee", "rknee", "lwrist",
    "rwrist", "lankle", "rankle",
)

TC_SENSOR_NAMES = (
    "head", "sternum", "pelvis", "left_upper_arm", "right_upper_arm",
    "left_forearm", "right_forearm", "left_thigh", "right_thigh",
    "left_shank", "right_shank", "left_foot", "right_foot",
)

DIP_CONVENTIONS = {
    "fps": 60.0, "acceleration": "world_linear_m_s2",
    "orientation": "segment_to_world", "pose": "local_axis_angle_radians",
    "status": "user_approved_segment_mapping_with_chest_proxy",
}

TC_CONVENTIONS = {
    "fps": 60.0, "quaternion_sensors": "wxyz", "quaternion_calibration": "xyzw",
    "acceleration": "sensor_specific_force_m_s2", "gravity_world": list(SETTINGS.tc_gravity),
    "world_conversion": "rotate_Y_pi_after_reference_calibration",
    "orientation_calibration": "first_reference_pose_known_pose",
    "vicon_input_unit": "inches", "vicon_world_conversion": "negate_x_and_z",
    "status": "declared_reference_reader_convention_logged_per_sequence",
}

DIP_MAPPING = {
    0: MappingEntry(2, "Belly is the pelvis reference in the TransPose six-sensor convention.", "reference"),
    1: MappingEntry(9, "Hip rotation describes the left thigh; use lhip.", "draft"),
    2: MappingEntry(10, "Hip rotation describes the right thigh; use rhip.", "draft"),
    4: MappingEntry(11, "Knee rotation describes the left shank; TransPose uses lknee.", "reference"),
    5: MappingEntry(12, "Knee rotation describes the right shank; TransPose uses rknee.", "reference"),
    7: MappingEntry(15, "Ankle rotation describes the left foot; use lankle.", "draft"),
    8: MappingEntry(16, "Ankle rotation describes the right foot; use rankle.", "draft"),
    9: MappingEntry(3, "Spine3 uses the left chest as an asymmetric upper-torso proxy; explicit review required.", "proxy"),
    15: MappingEntry(0, "Head sensor in the DIP/TransPose convention.", "reference"),
    16: MappingEntry(5, "Shoulder rotation describes the left upper arm; use lshoulder.", "draft"),
    17: MappingEntry(6, "Shoulder rotation describes the right upper arm; use rshoulder.", "draft"),
    18: MappingEntry(7, "Elbow rotation describes the left forearm; TransPose uses lelbow.", "reference"),
    19: MappingEntry(8, "Elbow rotation describes the right forearm; TransPose uses relbow.", "reference"),
}

TC_MAPPING = {
    0: MappingEntry(2, "Pelvis in dip18 ID_SMPL_0.", "reference"),
    1: MappingEntry(7, "Left thigh corresponds to SMPL hip, ID_SMPL_0.", "reference"),
    2: MappingEntry(8, "Right thigh corresponds to SMPL hip, ID_SMPL_0.", "reference"),
    4: MappingEntry(9, "Left shank corresponds to SMPL knee, ID_SMPL_0.", "reference"),
    5: MappingEntry(10, "Right shank corresponds to SMPL knee, ID_SMPL_0.", "reference"),
    7: MappingEntry(11, "Left foot stream after the eleven dip18 body streams; verify actual file label.", "draft"),
    8: MappingEntry(12, "Right foot stream after the eleven dip18 body streams; verify actual file label.", "draft"),
    9: MappingEntry(1, "Sternum corresponds to Spine3 in ID_SMPL_0.", "reference"),
    15: MappingEntry(0, "Head corresponds to joint 15 in ID_SMPL_0.", "reference"),
    16: MappingEntry(3, "Left upper arm corresponds to shoulder 16 in ID_SMPL_0.", "reference"),
    17: MappingEntry(4, "Right upper arm corresponds to shoulder 17 in ID_SMPL_0.", "reference"),
    18: MappingEntry(5, "Left forearm corresponds to elbow 18 in ID_SMPL_0.", "reference"),
    19: MappingEntry(6, "Right forearm corresponds to elbow 19 in ID_SMPL_0.", "reference"),
}

TC_LABEL_ALIASES = {
    "head": 0, "sternum": 1, "spine": 1, "pelvis": 2, "hips": 2, "waist": 2,
    "leftupperarm": 3, "leftarm": 3, "lupperarm": 3, "luparm": 3,
    "rightupperarm": 4, "rightarm": 4, "rupperarm": 4, "ruparm": 4,
    "leftforearm": 5, "leftlowerarm": 5, "lforearm": 5, "llowarm": 5,
    "rightforearm": 6, "rightlowerarm": 6, "rforearm": 6, "rlowarm": 6,
    "leftthigh": 7, "leftupperleg": 7, "leftupleg": 7, "lthigh": 7, "lupleg": 7,
    "rightthigh": 8, "rightupperleg": 8, "rightupleg": 8, "rthigh": 8, "rupleg": 8,
    "leftshank": 9, "leftlowerleg": 9, "leftleg": 9, "lshank": 9, "llowleg": 9,
    "rightshank": 10, "rightlowerleg": 10, "rightleg": 10, "rshank": 10, "rlowleg": 10,
    "leftfoot": 11, "lfoot": 11, "rightfoot": 12, "rfoot": 12,
}

def require_approval():
    if not APPROVED:raise RuntimeError("Mapping approval is disabled. Review README.md section 4 and src/mapping.py, then set mapping_acknowledged in src/config.py only after review.")

def mapping_table(dataset):
    if dataset=='dip':return DIP_MAPPING
    if dataset=='totalcapture':return TC_MAPPING
    raise ValueError("Dataset must be dip or totalcapture.")

def validate_mapping():
    for dataset,names in (('dip',DIP_SENSOR_NAMES),('totalcapture',TC_SENSOR_NAMES)):
        table=mapping_table(dataset)
        if set(table)!=set(CANDIDATE_POOL):raise ValueError(f'{dataset}: mapping must contain exactly the reviewed 13 sites.')
        indices=[]
        for entry in table.values():
            if type(entry.sensor) is not int or not 0<=entry.sensor<len(names) or not entry.reason:
                raise ValueError(f'{dataset}: invalid sensor mapping.')
            indices.append(entry.sensor)
        if len(set(indices))!=len(CANDIDATE_POOL):raise ValueError(f'{dataset}: the common pool must map to distinct streams.')

def realizability(dataset,joints,*,require_six=True):
    validate_mapping();chosen=joint_set(joints)
    valid_size=len(chosen)==6 or chosen==CANDIDATE_POOL
    if require_six and not valid_size:raise ValueError('Real evaluation accepts six sites or exactly the all-13 reference.')
    table=mapping_table(dataset);names=DIP_SENSOR_NAMES if dataset=='dip' else TC_SENSOR_NAMES
    unavailable=[j for j in chosen if j not in table or table[j].sensor is None]
    sensors=[table[j].sensor if j in table else None for j in chosen]
    collisions={names[s]:[j for j,v in zip(chosen,sensors) if v==s] for s in sensors if s is not None and sensors.count(s)>1}
    return {'dataset':dataset,'joints':list(chosen),'sensor_indices':sensors,
        'realizable':valid_size and chosen[0]==0 and not unavailable and not collisions,
        'unavailable_joints':unavailable,'collisions':collisions,'mapping_approved':APPROVED,
        'placement_qualifications':[{'joint':j,'name':JOINT_NAMES[j],'sensor':table[j].sensor,
            'confidence':table[j].confidence,'reason':table[j].reason} for j in chosen if j in table and table[j].confidence!='reference'],
        'qualification':'User approved segment correspondences; DIP Spine3 uses left chest as a placement proxy. Exact mounting equivalence is not established.'}

def sensor_indices(dataset,joints):
    require_approval();report=realizability(dataset,joints)
    if not report['realizable']:raise ValueError(f'Non-realizable configuration: {report}')
    return tuple(report['sensor_indices'])

def identity():
    return {'approved':APPROVED,'pool':CANDIDATE_POOL,'dip_sensor_names':DIP_SENSOR_NAMES,'tc_sensor_names':TC_SENSOR_NAMES,
        'dip':{str(j):asdict(e) for j,e in DIP_MAPPING.items()},'totalcapture':{str(j):asdict(e) for j,e in TC_MAPPING.items()},
        'dip_conventions':DIP_CONVENTIONS,'tc_conventions':TC_CONVENTIONS,'tc_label_aliases':TC_LABEL_ALIASES}

def mapping_fingerprint():return fingerprint(identity())

def baseline_report():
    return {dataset:{**{name:realizability(dataset,sites) for name,sites in BASELINES.items()},
            'upper_13':realizability(dataset,CANDIDATE_POOL)} for dataset in ('dip','totalcapture')}

"""Edit this file, then run a numbered script. No command-line flags are used.

Paths are relative to this project, never to the PowerShell working directory.
Changing the empirical tolerance does not invalidate synthetic data or models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import hashlib
import json
import math
import time
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
ACTIVE_PRESET = "full"                 # "pilot" for a tiny end-to-end rehearsal
DATA_ROOT = PROJECT / "data"
CACHE_ROOT = PROJECT / "cache"
RUN_ROOT = PROJECT / "run"
AMASS_DIRECTORY = DATA_ROOT / "amass" / "CMU"
SMPL_FILE = DATA_ROOT / "smpl" / "SMPL_male.pkl"  # numeric .npz is also accepted
DIP_DIRECTORY = DATA_ROOT / "dip_imu"
TOTALCAPTURE_DIRECTORY = DATA_ROOT / "totalcapture"
TOTALCAPTURE_SMPL_DIRECTORY = DATA_ROOT / "totalcapture_smpl"  # flat s1_acting1.pkl accepted

# Set ONLY after inspecting the output of 5_greedy_a.py. None deliberately blocks B.
# Examples of accepted syntax: 0.5 or [0.25, 0.5, 1.0]. Values are degrees.
GREEDY_TOLERANCES_DEG = None
MAIN_TOLERANCE_DEG = None              # None = first supplied tolerance

IMUOPTIMIZE_COMMIT = "9ec6dc40731d230d948df2575b5fa4d39d7b1941"
PIPELINE_VERSION = "common13-pelvis-fixed-swap-v1"
CANDIDATE_POOL = (0,1,2,4,5,7,8,9,15,16,17,18,19)
# Original SMPL indices are retained; synthesis and pose output still have 24 joints.

ARCHITECTURES = ("birnn",)
JOINT_NAMES = ("Pelvis", "L Hip", "R Hip", "Spine1", "L Knee", "R Knee", "Spine2",
               "L Ankle", "R Ankle", "Spine3", "L Foot", "R Foot", "Neck", "L Collar",
               "R Collar", "Head", "L Shoulder", "R Shoulder", "L Elbow", "R Elbow",
               "L Wrist", "R Wrist", "L Hand", "R Hand")
PARENTS = (-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21)
SIP_ERROR_JOINTS = (1,2,16,17)
SIP_DIP = (0,4,5,15,18,19)
BASELINES = {"sip_dip": SIP_DIP}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False,
                      default=lambda x: str(x) if isinstance(x,Path) else x.tolist() if hasattr(x,"tolist") else x.item() if hasattr(x,"item") else (_ for _ in ()).throw(TypeError(type(x).__name__)))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def joint_set(values, *, size=None):
    values=tuple(values)
    if not values or any(type(j) is not int or not 0<=j<24 for j in values) or len(set(values))!=len(values):
        raise ValueError("Sites must be distinct integer SMPL indices 0–23.")
    if size is not None and len(values)!=size: raise ValueError(f"Expected {size} sites.")
    return tuple(sorted(values))


def tolerance_values(value=None):
    value = GREEDY_TOLERANCES_DEG if value is None else value
    if value is None:
        raise ValueError("Review run/<preset>/greedy_a_gap_review.csv, then enter GREEDY_TOLERANCES_DEG in src/config.py as a number or a list of degree values. Run 5_greedy_a.py first.")
    values = [value] if isinstance(value,(int,float)) and not isinstance(value,bool) else value
    if not isinstance(values,(list,tuple)) or not values:
        raise ValueError("GREEDY_TOLERANCES_DEG must be a number or nonempty list/tuple.")
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError("Each tolerance must be finite and nonnegative degrees.")
    result=tuple(dict.fromkeys(float(v) for v in values))
    if MAIN_TOLERANCE_DEG is not None and (isinstance(MAIN_TOLERANCE_DEG,bool) or not isinstance(MAIN_TOLERANCE_DEG,(int,float)) or float(MAIN_TOLERANCE_DEG) not in result):
        raise ValueError("MAIN_TOLERANCE_DEG must occur in GREEDY_TOLERANCES_DEG.")
    return result


def tolerance_label(value):
    number=float(value)
    text=str(int(number)) if number.is_integer() and abs(number)<1e16 else repr(number)
    return "greedy_b_tau_"+text.replace(".","p").replace("-","m").replace("+","")


def main_tolerance():
    values=tolerance_values()
    return values[0] if MAIN_TOLERANCE_DEG is None else float(MAIN_TOLERANCE_DEG)


@dataclass(frozen=True)
class ModelSpec:
    rnn_hidden: int = 64
    rnn_layers: int = 2
    rnn_bidirectional: bool = True
    rnn_dropout: float = 0.0
    input_channels: int = 288          # structural constant: 24 × 12
    output_channels: int = 216         # structural constant: 24 × 9
    max_window_length: int = 5000


@dataclass(frozen=True)
class TrainingBudget:
    stage: str
    epochs: int
    train_fraction: float
    max_train_subjects: int | None
    max_sequences_per_subject: int | None
    max_frames_per_sequence: int | None
    window_length: int
    stride: int
    effective_batch_size: int
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float
    loss: str
    precision: str
    num_workers: int
    model: ModelSpec

    @property
    def identity(self):
        value=asdict(self);value.pop("stage");return value
    @property
    def id(self):return fingerprint(self.identity)


@dataclass(frozen=True)
class Settings:
    name: str = "full"
    device: str = "cuda"              # "cpu" or "auto" also accepted
    max_minutes: float | None = None  # graceful per-script time limit
    search_epochs: int = 3
    final_epochs: int = 5
    search_train_fraction: float = 1.0
    final_train_fraction: float = 1.0
    batch_size: int = 8
    window_length: int = 60
    stride: int = 60
    num_workers: int = 0              # Windows-safe default
    amp: bool = False
    optimizer: str = "Adam"            # SGD or Adam; configure lr appropriately
    learning_rate: float = 0.001
    momentum: float = 0.0
    weight_decay: float = 0.0
    loss: str = "matrix_mse"
    model: ModelSpec = field(default_factory=ModelSpec)
    seed: int = 1234
    enable_additional_final_seeds: bool = False
    additional_final_seeds: tuple[int,...] = (1,42)
    random_draws: int = 30
    swap_max_accepted: int | None = None
    swap_min_improvement_deg: float = 0.0
    benchmark_swap_sweeps_per_path: int = 2  # scenario, not a guaranteed bound
    sensor_budget: int = 6
    train_subject_fraction: float = 0.8
    max_train_subjects: int | None = None
    max_val_subjects: int | None = None
    max_sequences_per_subject: int | None = None
    max_frames_per_sequence: int | None = None
    fps: float = 60.0
    smooth_n: int = 4
    acceleration_scale: float = 30.0
    fk_chunk_frames: int = 64
    metric_chunk_frames: int = 32
    checkpoint_interval_seconds: float = 60.0
    prediction_interval_seconds: float = 60.0
    sequence_cache_mib: int = 512
    mic_samples: int = 1024
    mic_alpha: float = 0.6
    mic_clumps: float = 15.0
    mic_estimator: str = "mic_approx"
    mic_checkpoint_pairs: int = 10
    dip_enabled: bool = True
    totalcapture_enabled: bool = True
    dip_test_subjects: tuple[str,...] = ("s_09","s_10")
    tc_subjects: tuple[str,...] = ("s1","s2","s3","s4","s5")
    tc_excluded: tuple[tuple[str,str],...] = (("s5","acting3"),)
    mapping_acknowledged: bool = False  # execution consent; mapping remains provisional
    tc_allow_numeric_order: bool = False # enable only after independently confirming numeric sensor order
    tc_first_reference_pose_calibration: bool = True
    tc_gravity: tuple[float,float,float] = (0.0,9.8707,0.0)
    dip_fill_passes: int = 4
    dip_trim_frames: int = 6
    benchmark_batch_sizes: tuple[int,...] = (8,)
    benchmark_epochs: int = 1
    benchmark_target_hours: float = 72.0
    dpi: int = 180
    grid_page_size: int = 4
    pose_reference_label: str = "sip_dip"
    random_pose_examples: int = 3
    random_pose_configurations: tuple[str,...] = ("greedy_a","greedy_a_swap","sip_dip")
    make_synthetic_pose_examples: bool = True
    camera_elevation: float = 12.0
    camera_azimuth: float = -65.0
    threshold_review_degrees: tuple[float,...] = (0.1,0.25,0.5,0.75,1.0)

    @property
    def split_seed(self):return self.seed
    @property
    def random_seed(self):return self.seed
    @property
    def search_seed(self):return self.seed
    @property
    def mic_seed(self):return self.seed
    @property
    def final_seeds(self):return (self.seed, *self.additional_final_seeds) if self.enable_additional_final_seeds else (self.seed,)
    @property
    def identity(self):return asdict(self)
    @property
    def datasets(self):return tuple(k for k,on in (("dip",self.dip_enabled),("totalcapture",self.totalcapture_enabled)) if on)

    def validate(self):
        if len(CANDIDATE_POOL)!=13 or tuple(sorted(set(CANDIDATE_POOL)))!=CANDIDATE_POOL or CANDIDATE_POOL[0]!=0 or any(type(j) is not int or not 0<=j<24 for j in CANDIDATE_POOL):
            raise ValueError("Expected the reviewed 13-site pool with original SMPL indices and mandatory pelvis.")
        if any(not set(sites)<=set(CANDIDATE_POOL) for sites in BASELINES.values()):raise ValueError("Baseline outside reviewed candidate pool.")
        if self.swap_max_accepted is not None and (type(self.swap_max_accepted) is not int or self.swap_max_accepted<1):raise ValueError("swap_max_accepted must be None or a positive integer.")
        if isinstance(self.swap_min_improvement_deg,bool) or not math.isfinite(self.swap_min_improvement_deg) or self.swap_min_improvement_deg<0:raise ValueError("Invalid swap_min_improvement_deg.")
        if type(self.benchmark_swap_sweeps_per_path) is not int or self.benchmark_swap_sweeps_per_path<1:raise ValueError("Benchmark swap scenario must have positive sweep count.")
        for value in (self.search_epochs,self.final_epochs,self.batch_size,self.window_length,self.stride,self.random_draws,self.mic_samples,self.fk_chunk_frames,self.metric_chunk_frames,self.model.rnn_hidden,self.model.rnn_layers,self.mic_checkpoint_pairs,self.benchmark_epochs,self.dip_fill_passes,self.dip_trim_frames):
            if type(value) is not int or value<1:raise ValueError("Counts, epochs and model dimensions must be positive integers.")
        if self.device not in ("cpu","cuda","auto"):raise ValueError("device must be cpu, cuda or auto.")
        if any(type(v) is not int or v<1 for v in self.benchmark_batch_sizes) or not self.benchmark_batch_sizes:raise ValueError("benchmark_batch_sizes must contain positive integers.")
        if any(not math.isfinite(v) for v in (self.momentum,self.weight_decay,self.mic_clumps,self.mic_alpha,self.benchmark_target_hours)) or self.momentum<0 or self.benchmark_target_hours<=0:raise ValueError("Invalid optimizer/MIC/benchmark numbers.")
        if len(self.tc_gravity)!=3 or not all(math.isfinite(v) for v in self.tc_gravity):raise ValueError("tc_gravity must have three finite values.")
        if any(not math.isfinite(v) or v<0 for v in self.threshold_review_degrees):raise ValueError("Threshold review values must be nonnegative finite degrees.")
        if not self.threshold_review_degrees:raise ValueError("Supply at least one threshold review value.")
        if self.name=="full" and self.search_epochs>=self.final_epochs:raise ValueError("Use fewer search epochs than final epochs.")
        if self.seed!=1234 or self.additional_final_seeds!=(1,42):raise ValueError("Main seed is 1234; optional extra training seeds are 1 and 42.")
        if self.num_workers!=0:raise ValueError("This Windows implementation requires num_workers=0.")
        if not 0<self.search_train_fraction<=self.final_train_fraction<=1 or not 0<self.train_subject_fraction<1:raise ValueError("Invalid train/window fractions.")
        if not 1<=self.stride<=self.window_length<=self.model.max_window_length:raise ValueError("Require stride <= window <= model limit.")
        if self.sensor_budget!=6 or self.fps!=60 or self.smooth_n!=4 or self.acceleration_scale!=30:raise ValueError("Supported methodology: six sites, 60 Hz, smooth_n=4, acceleration scale=30.")
        if self.model.rnn_bidirectional is not True:raise ValueError("This project implements BiRNN; bidirectionality must remain True.")
        if self.model.input_channels!=288 or self.model.output_channels!=216:raise ValueError("Structural dimensions must remain 288 input and 216 output channels.")
        if not 0<=self.model.rnn_dropout<1 or self.model.rnn_layers==1 and self.model.rnn_dropout!=0:raise ValueError("Invalid recurrent dropout.")
        if self.optimizer not in ("SGD","Adam") or self.loss!="matrix_mse":raise ValueError("Use SGD/Adam and matrix_mse.")
        if not math.isfinite(self.learning_rate) or self.learning_rate<=0 or self.weight_decay<0:raise ValueError("Invalid optimizer settings.")
        if self.optimizer=="Adam" and self.momentum!=0:raise ValueError("momentum is an SGD setting; use 0 with Adam.")
        if not 0<self.mic_alpha<=1 or self.mic_clumps<=0 or self.mic_estimator not in ("mic_approx","mic_e"):raise ValueError("Invalid minepy settings.")
        if self.random_pose_examples<0 or self.sequence_cache_mib<2 or not 40<=self.dpi<=300 or not 1<=self.grid_page_size<=12:raise ValueError("Invalid cache/figure settings.")
        for seconds in (self.checkpoint_interval_seconds,self.prediction_interval_seconds):
            if not math.isfinite(seconds) or seconds<=0:raise ValueError("Checkpoint intervals must be positive.")
        for cap in (self.max_train_subjects,self.max_val_subjects,self.max_sequences_per_subject,self.max_frames_per_sequence):
            if cap is not None and (type(cap) is not int or cap<1):raise ValueError("Optional data caps must be positive integers.")

    def budget(self,stage):
        if stage not in ("search","final"):raise ValueError("Unknown training stage.")
        return TrainingBudget(stage,self.search_epochs if stage=="search" else self.final_epochs,
            self.search_train_fraction if stage=="search" else self.final_train_fraction,
            self.max_train_subjects,self.max_sequences_per_subject,self.max_frames_per_sequence,
            self.window_length,self.stride,self.batch_size,self.optimizer,self.learning_rate,self.momentum,
            self.weight_decay,self.loss,"cuda_amp_float16_else_float32" if self.amp else "float32",self.num_workers,self.model)


FULL = Settings()
PILOT = replace(FULL,name="pilot",search_epochs=1,final_epochs=1,max_train_subjects=2,
    max_val_subjects=1,max_sequences_per_subject=1,max_frames_per_sequence=64,
    window_length=16,stride=16,mic_samples=32,random_pose_examples=1)
SETTINGS = {"full":FULL,"pilot":PILOT}[ACTIVE_PRESET]


class StopRequested(Exception):pass


class Deadline:
    def __init__(self,max_minutes):
        if max_minutes is not None and (not math.isfinite(max_minutes) or max_minutes<=0):raise ValueError("max_minutes must be positive or None.")
        self.started=time.monotonic();self.ends=None if max_minutes is None else self.started+max_minutes*60
    @property
    def expired(self):return self.ends is not None and time.monotonic()>=self.ends
    def check(self):
        if self.expired:raise StopRequested("Time limit reached. Run the same script to resume.")


@dataclass(frozen=True)
class Paths:
    project: Path
    data: Path
    cache: Path
    runs: Path
    @property
    def amass(self):return AMASS_DIRECTORY if self.data==DATA_ROOT else self.data/"amass"/"CMU"
    @property
    def smpl(self):
        p=SMPL_FILE if self.data==DATA_ROOT else self.data/"smpl"/"SMPL_male.pkl"
        return p.with_suffix(".npz") if p.suffix==".pkl" and not p.is_file() and p.with_suffix(".npz").is_file() else p
    @property
    def dip(self):return DIP_DIRECTORY if self.data==DATA_ROOT else self.data/"dip_imu"
    @property
    def totalcapture(self):return TOTALCAPTURE_DIRECTORY if self.data==DATA_ROOT else self.data/"totalcapture"
    @property
    def totalcapture_smpl(self):return TOTALCAPTURE_SMPL_DIRECTORY if self.data==DATA_ROOT else self.data/"totalcapture_smpl"
    @property
    def vendor(self):return self.project/"third_party"/"imuoptimize"
    @property
    def ledger(self):return self.runs/"training_ledger.jsonl"
    @property
    def figures(self):return self.runs/"figures"

# Optimizing Sensor Placement for Sparse-IMU Human Pose Estimation via Combinatorial Search

This repository contains the implementation for the paper **Optimizing Sensor Placement for Sparse-IMU Human Pose Estimation via Combinatorial Search**.

**Paper:** link TBA.

The experiments select six IMU locations from a common pool of thirteen, with the pelvis mandatory for normalization. The pipeline compares thirty random configurations, error-only greedy selection (A), greedy selection with an MIC tie-breaker within an empirical error tolerance (B), and single-sensor swap refinement of both greedy results. Models are trained on synthetic CMU AMASS data and evaluated on DIP-IMU and TotalCapture.

This is a code-only release. Datasets, SMPL models, training checkpoints, cached sequences, and experimental outputs must be obtained or generated separately.

## Repository structure

| Path | Contents |
|---|---|
| `src/config.py` | Paths, presets, training budgets, model settings, tolerance, and evaluation options |
| `src/mapping.py` | The thirteen candidate sites and DIP-IMU/TotalCapture correspondences |
| `src/0_check.py`–`src/11_visualizations.py` | Numbered experiment steps |
| `src/benchmark.py` | Hardware timing and protocol runtime estimates |
| `src/shared/` | Data readers, synthesis adapter, models, training, search, metrics, analysis, and plotting |
| `tools/install_minepy.py` | Windows/Python 3.11 minepy build helper |
| `tools/export_numeric_smpl.py` | Optional conversion of a licensed legacy SMPL model to numeric arrays |
| `tools/static_review.py` | Static source and third-party integrity checks |
| `install_minepy.ps1` | PowerShell entry point for the minepy installer |
| `selftest.py` | Tests using generated data |
| `requirements.txt` | Pinned Python dependencies; install CUDA PyTorch separately |
| `third_party/` | Initially empty; populated locally using the instructions below |
| `data/` | Local datasets and licensed body model; not distributed |
| `cache/` | Generated synthetic data and derived training arrays; not distributed |
| `run/` | Generated reports, logs, models, predictions, tables, and figures; not distributed |

Git does not retain empty directories. The setup commands create the required directories locally. Place the contents of this repository directly inside one project directory; an additional nested `imuplace13/imuplace13/` wrapper is unnecessary.

## 1. Install the environment

The reference environment is Windows, Python 3.11.9, PyTorch 2.5.1 with CUDA 12.1, and an NVIDIA RTX 3050 Ti Laptop GPU with 4 GB VRAM. A different machine may produce different timings or small numerical differences.

Install Python 3.11, Git, and Microsoft C++ Build Tools with the C++ desktop workload and a Windows SDK. The compiler is needed when building minepy from source.

Open PowerShell in the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "pip==24.3.1" "setuptools==75.3.0" "wheel==0.44.0"
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
python .\tools\install_minepy.py
python -m pip install -r .\requirements.txt
New-Item -ItemType Directory -Force data, cache, run, third_party | Out-Null
```

The minepy helper downloads the official 1.2.6 source distribution, verifies its published checksum, regenerates the old Cython wrapper for Python 3.11, and builds a local wheel. It records the build under `wheels/`. The project requires native minepy; no alternative MIC approximation is substituted. If an existing environment already has a working minepy 1.2.6 installation, the build step can be skipped.

If PowerShell prevents environment activation, use `.\.venv\Scripts\python.exe` in place of `python` in the commands below.

Check the installation:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import numpy as np; from minepy import MINE; m=MINE(); x=np.arange(32,dtype=float); m.compute_score(x,x); print('MIC:',m.mic())"
```

The native minepy estimator called `mic_approx` is part of minepy itself; its name does not indicate a fallback implementation in this repository.

## 2. Obtain the third-party code

The synthesis and forward-kinematics adapter imports selected components from [IMUOptimize](https://github.com/varun-ramani/IMUOptimize), by Varun Ramani and contributors. The required commit is:

```text
9ec6dc40731d230d948df2575b5fa4d39d7b1941
```

No explicit source-code license grant was found in the audited copy of this commit. The upstream source is therefore not redistributed here. Obtain any permission needed from its authors for your intended use; attribution alone does not grant permission. Do not assume that this repository's license, if provided, covers IMUOptimize, minepy, datasets, or SMPL models.

From the repository root, download the pinned source without Windows line-ending conversion:

```powershell
git -c core.autocrlf=false clone --no-checkout https://github.com/varun-ramani/IMUOptimize.git .\third_party\imuoptimize
git -C .\third_party\imuoptimize -c core.autocrlf=false checkout --detach 9ec6dc40731d230d948df2575b5fa4d39d7b1941
```

The adapter also requires a checksum manifest. Run this PowerShell block from the repository root after checkout:

```powershell
@'
from pathlib import Path
import hashlib
import json
import subprocess

root = Path("third_party/imuoptimize").resolve()
commit = "9ec6dc40731d230d948df2575b5fa4d39d7b1941"
def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args])
if git("rev-parse", "HEAD").decode().strip() != commit:
    raise SystemExit("Incorrect IMUOptimize commit.")
paths = git("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")
hashes = {}
for relative in filter(None, paths):
    committed = git("cat-file", "blob", f"{commit}:{relative}")
    local = root / relative
    if not local.is_file() or local.read_bytes() != committed:
        raise SystemExit(f"Upstream file differs from the pinned commit: {relative}")
    hashes[relative] = hashlib.sha256(committed).hexdigest()
manifest = root.parent / "imuoptimize.manifest.json"
manifest.write_text(json.dumps({"commit": commit, "files_sha256": hashes},
                              indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("Verified", len(hashes), "files; wrote", manifest)
'@ | python -
```

Keep the downloaded files unmodified. Compatibility handling belongs in `src/shared/vendor_adapter.py`. The adapter isolates upstream logging utilities; do not run the upstream shell scripts as part of this workflow. Do not install the upstream project's separate training environment or its Transformer dependencies.

The manifest is generated locally and must remain available at runtime, even if `third_party/` is empty in the public repository. The GPL-licensed dip18 project was used as a format reference only; it does not need to be downloaded to run this code.

## 3. Download and arrange the data

Use the official portals and accept their applicable access terms. Downloads requiring an account are not automated. Do not upload datasets, body-model files, access credentials, or derived restricted data to this repository.

### CMU AMASS: synthetic training and validation

1. Register or sign in at [AMASS](https://amass.is.tue.mpg.de/).
2. Download the **CMU subset in SMPL+H format**. The reader expects `*_poses.npz` files with `poses[T,156]`, `trans[T,3]`, `betas` with at least ten coefficients, and `mocap_framerate`.
3. Extract the subject folders under `data/amass/CMU/`. If the archive adds another wrapper directory, adjust `AMASS_DIRECTORY` in `src/config.py`.
4. Preserve the original subject/release directory names, including the Justin/Rory folders numbered 18–23. The code groups these related releases conservatively for the train/validation split.

The pipeline retains the upstream 59/60/120 Hz handling; 120 Hz is downsampled by two. Retained 59 Hz recordings generate a warning because synthetic derivatives use the 60 Hz convention. Unsupported or short sequences are reported during synthesis.

### SMPL: synthesis, metrics, and mesh figures

1. Register at [SMPL](https://smpl.is.tue.mpg.de/).
2. Obtain the original **24-joint male SMPL model**. This is separate from the SMPL+H motion parameters in AMASS. SMPL-X is not a compatible replacement.
3. Place the numeric model at `data/smpl/SMPL_male.pkl`, or set `SMPL_FILE` to its actual location. A numeric `.npz` is also supported.

Required fields are `J_regressor`, `weights`, `posedirs`, `shapedirs`, `v_template`, `J`, `f`, and `kintree_table`. The data check validates their dimensions and the standard 24-joint hierarchy.

If a legacy pickle depends on Chumpy, use a licensed environment that can already load it. Edit `SOURCE` and `OUTPUT` in `tools/export_numeric_smpl.py`, run that converter in the legacy environment, then set `SMPL_FILE` to the resulting numeric archive. The main environment does not require Chumpy. Load only trusted pickle files.

### DIP-IMU: real-data evaluation

1. Register or sign in at [Deep Inertial Poser](https://dip.is.tuebingen.mpg.de/) and open Download.
2. Obtain the **original raw 17-sensor DIP-IMU dataset**, not the normalized six-sensor TensorFlow inputs.
3. Extract the subject folders under `data/dip_imu/`. The default evaluation uses `s_09` and `s_10`.

The required fields are `gt[T,72]`, `imu_acc[T,17,3]`, and `imu_ori[T,17,3,3]`. The readers support Latin-1 decoding for older pickles. Reference poses are SIP-derived estimates, not independent optical ground truth. DIP-IMU is a separate recording dataset from TotalCapture.

### TotalCapture: real-data evaluation

1. Request access through the [TotalCapture dataset portal](https://cvssp.org/data/totalcapture/).
2. From the IMU section, download the original recordings for subjects 1–5, including `.sensors` files and the corresponding `_calib_imu_ref.txt` and `_calib_imu_bone.txt` files.
3. Extract them under `data/totalcapture/`, retaining the supplied filenames and calibration files. The reader searches recursively.
4. From the DIP download portal, obtain the corresponding TotalCapture SMPL/MoSh reference parameters. Place them under `data/totalcapture_smpl/`. Flat filenames such as `s1_acting1.pkl` are accepted; subject subfolders are not required.
5. The primary SMPL evaluation does not require videos, camera calibration, mattes, gyroscope/magnetometer archives, raw C3D, or BVH. Optional Vicon position files are used only for separate diagnostics. The reader supports the documented named, tab-separated `gt_skel_gbl_pos.txt` format, not arbitrary CSV/BVH exports.

Example files for one recording:

```text
data/totalcapture/s1/s1_acting1_Xsens.sensors
data/totalcapture/s1/s1_acting1_calib_imu_ref.txt
data/totalcapture/s1/s1_acting1_calib_imu_bone.txt
data/totalcapture_smpl/s1_acting1.pkl
```

The default protocol considers subjects 1–5 and excludes `s5/acting3`. The reader records discovered recordings, reference offsets, exclusions, and usable frame counts. Its orientation convention uses the first paired SMPL reference pose for calibration and excludes that frame from scoring. Both calibration files are checked, although bone calibration is not applied under this selected convention. This is known-pose calibration, not calibration-free evaluation. Trimming unmatched lengths alone does not establish synchronization.

If either real dataset is unavailable, disable it explicitly in `src/config.py`. Synthetic training and selection can proceed without it; reproducing both real-data tables requires both datasets.

### Expected local layout

```text
data/
  amass/CMU/01/01_01_poses.npz
  smpl/SMPL_male.pkl
  dip_imu/s_09/*.pkl
  dip_imu/s_10/*.pkl
  totalcapture/.../*_Xsens.sensors
  totalcapture/.../*_calib_imu_ref.txt
  totalcapture/.../*_calib_imu_bone.txt
  totalcapture_smpl/s1_acting1.pkl
```

## 4. Configure the experiment

All routine settings are in `src/config.py`; commands do not require PowerShell flag arrays. Paths are based on the project directory rather than the current working directory.

Set the path constants first:

```python
DATA_ROOT = PROJECT / "data"
CACHE_ROOT = PROJECT / "cache"
RUN_ROOT = PROJECT / "run"
AMASS_DIRECTORY = DATA_ROOT / "amass" / "CMU"
SMPL_FILE = DATA_ROOT / "smpl" / "SMPL_male.pkl"
DIP_DIRECTORY = DATA_ROOT / "dip_imu"
TOTALCAPTURE_DIRECTORY = DATA_ROOT / "totalcapture"
TOTALCAPTURE_SMPL_DIRECTORY = DATA_ROOT / "totalcapture_smpl"
```

Absolute `Path` values may be used for data stored elsewhere. Edit the existing definitions rather than appending duplicate settings.

The full experiment settings are:

| Setting | Value |
|---|---|
| Candidate pool | `(0,1,2,4,5,7,8,9,15,16,17,18,19)` |
| Sensor budget | Six, including mandatory pelvis |
| Random configurations | 30 uniform draws; identical layouts reuse training |
| Model | Two-layer bidirectional LSTM, 64 hidden units per direction |
| Input/output | 288 zero-masked channels / 216 values for 24 local rotations |
| Optimizer | Adam, learning rate 0.001 |
| Precision | FP32 (`amp=False`) |
| Batch/window/stride | 8 / 60 frames / 60 frames |
| Search and swap budget | Three epochs, all retained training windows |
| Final budget | Five epochs from initialization, all retained training windows |
| Main seed | 1234 |
| Additional final seeds | 1 and 42, disabled by default |
| Paper tolerance | 0.25 degrees |
| Swap stopping | No accepted-swap cap; strictly improving moves only |

Set `ACTIVE_PRESET = "pilot"` for a small rehearsal. PILOT retains the same selection logic but limits the data and uses one epoch. Set `ACTIVE_PRESET = "full"` for the paper protocol. PILOT is a software rehearsal, not a substitute for the full experiment.

For the paper's tolerance:

```python
GREEDY_TOLERANCES_DEG = 0.25
MAIN_TOLERANCE_DEG = None
```

A list such as `[0.1, 0.25, 0.5]` runs multiple B variants and increases the workload; it does not reproduce the single-tolerance paper protocol. For a new tolerance study, finish Greedy A, inspect its validation error gaps, and fix the tolerance before examining real-data results. `None` blocks B until a value is supplied.

Within `Settings`, review `dip_enabled`, `totalcapture_enabled`, `device`, and `max_minutes`. A positive `max_minutes` sets a graceful limit per invocation; `None` removes it. Edit `ModelSpec` for architecture settings. Changing these settings creates a different experimental protocol; do not present it as an exact reproduction of the paper.

### Review sensor correspondence

Review `src/mapping.py` before real-data evaluation. The retained segment interpretations are:

| SMPL site | Segment | DIP-IMU stream | TotalCapture stream |
|---|---|---|---|
| 0 | Pelvis | 2: belly | 2: pelvis |
| 1 | Left thigh | 9: left hip | 7: left thigh |
| 2 | Right thigh | 10: right hip | 8: right thigh |
| 4 | Left shank | 11: left knee | 9: left shank |
| 5 | Right shank | 12: right knee | 10: right shank |
| 7 | Left foot | 15: left ankle | 11: left foot |
| 8 | Right foot | 16: right ankle | 12: right foot |
| 9 | Upper torso | 3: left chest, asymmetric proxy | 1: sternum |
| 15 | Head | 0: head | 0: head |
| 16 | Left upper arm | 5: left shoulder | 3: left upper arm |
| 17 | Right upper arm | 6: right shoulder | 4: right upper arm |
| 18 | Left forearm | 7: left elbow | 5: left forearm |
| 19 | Right forearm | 8: right elbow | 6: right forearm |

Stream indices are zero based. A correspondence does not establish identical sensor attachment coordinates. In particular, the DIP left-chest proxy is retained explicitly. Numeric TotalCapture sensor labels require independent verification before enabling `tc_allow_numeric_order`.

Mapping approval is controlled by `mapping_acknowledged` inside the `Settings` dataclass in `src/config.py`. `src/mapping.py` reads it as `APPROVED = SETTINGS.mapping_acknowledged`; edit the configuration value rather than overriding `APPROVED`. Restart a numbered script after changing the setting.

## 5. Check the installation and rehearse

After obtaining the third-party source:

```powershell
python .\tools\static_review.py
python .\selftest.py
```

The static checker uses `third_party/imuoptimize.manifest.json`; the generated-data self-test does not require the research datasets. Neither establishes physical calibration or correctness of the real recording metadata.

With `ACTIVE_PRESET = "pilot"`, follow the complete sequence below. Switch to `"full"` and repeat for the actual experiment. Outputs are separated by preset.

## 6. Run the experiment

Run one script at a time from the repository root.

### Check data, synthesize, and benchmark

```powershell
python .\src\0_check.py
python .\src\1_synthesize.py
python .\src\benchmark.py
```

- **0:** Check Python/CUDA/dependencies, SMPL, source files, enabled real datasets, and sensor mappings. Inspect `run/<preset>/environment_and_data_check.json` for missing files, shapes, and warnings.
- **1:** Generate and cache physical synthetic IMU signals and the capture-group train/validation split. All 24 sites are stored for compatibility with the synthesis pipeline, but only the thirteen candidate sites are selectable.
- **Benchmark:** Measure representative training, scoring, and MIC costs and project runtime for the current settings. Swap sweep counts are a scenario, not a guaranteed runtime. The benchmark does not automatically rewrite configuration settings.

### Train the references and run selection

```powershell
python .\src\2_upperbound.py
python .\src\3_baseline.py
python .\src\4_randoms.py
python .\src\5_greedy_a.py
```

- **2:** Train the all-thirteen-sensor reference using the final budget. Despite the historical filename, this is an empirical reference, not a guaranteed accuracy upper bound.
- **3:** Train the conventional SIP/DIP six-site baseline using the final budget.
- **4:** Evaluate thirty random six-site configurations using the search budget. Duplicate draws, if any, reuse training; they are not independent replications.
- **5:** Run error-only forward selection. Starting with pelvis, evaluate 12, 11, 10, 9, and 8 additions. Save the path, candidate scores, and `greedy_a_gap_review.csv` with a tolerance-review figure.

Inspect the A gap report before choosing an empirical tolerance for a new study. For paper reproduction, keep 0.25 degrees.

```powershell
python .\src\6_greedy_b.py
python .\src\7_swap.py
```

- **6:** Run B with the configured tolerance or tolerance list. A training-only 13-by-13 MIC matrix is computed once. Each sensor pair averages its 144 channel-pair scores. Within the allowed error band, B maximizes relevance minus redundancy; remaining ties use error and joint index.
- **7:** Refine A and every configured B using error-only single-sensor swaps. Keep pelvis and inspect all 35 replacements per sweep. Accept the lowest-error neighbor only when it strictly improves the search objective. Repeat until no improvement remains under the default settings.

Uncapped stopping after a complete non-improving sweep establishes a measured one-swap local optimum for the fixed search objective. It does not establish global optimality or guarantee better final-budget or real-data performance.

### Retrain, evaluate, analyze, and visualize

```powershell
python .\src\8_retrain.py
python .\src\9_evaluation.py
python .\src\10_analyze.py
python .\src\11_visualizations.py
```

- **8:** Train all final configurations from initialization for five epochs: thirty random labels, SIP/DIP, all thirteen, A, B, A + swap, and B + swap. One B tolerance gives 36 labels before deduplication. Identical subsets share a final model, and matching reference/baseline models from steps 2–3 are reused.
- **9:** Evaluate frozen synthetic-trained models on every enabled real dataset using the common mapped cohort. No fine-tuning or noise augmentation is applied.
- **10:** Export final metric tables and analysis reports, including random percentiles, selection overlap, tie-breaker statistics, and swap outcomes.
- **11:** Generate Agg PNG figures, including random distributions, greedy/swap curves, metric comparisons, sensor layouts, and best/worst/random pose examples. `pose_examples.csv` records actual frame errors and the example-selection rule.

## 7. Outputs and interpretation

Reports, logs, checkpoints, predictions, and tables are written under `run/<preset>/`; physical and derived arrays are written under `cache/<preset>/`. Figures are stored in fingerprinted directories under `run/<preset>/figures/`.

The search objective is frame-weighted mean local geodesic rotation error over all 24 joints. Final reporting uses equal-sequence mean metrics. Reported metrics are local rotation error, global rotation error, SIP-error, pelvis-aligned MPJPE, pelvis-aligned mesh error, and jitter. MPJPE is not used to select the greedy configurations.

Synthetic final reporting reuses the selection validation partition and is not an independent test. Random percentiles describe the thirty sampled layouts, not the complete 792-layout space. With one seed, no training-seed standard deviation, significance test, or confidence interval is reported.

By default, pose examples are selected using SIP/DIP: find its lowest/highest sequence MPJPE, then the lowest/highest frame MPJPE within those sequences. All models use the same selected frames. These are baseline-selected examples, not each method's own best/worst poses or the globally most extreme frames.


## 8. Resume and troubleshooting

Rerun the same script to resume durable progress. Models are reused only when their data, subset, seed, budget, and numerical identities match. Checkpoint and prediction progress are saved approximately every 60 seconds; a forced process termination may repeat work since the last save. Changing the tolerance does not by itself invalidate an identical trained subset model.

| Problem | Action |
|---|---|
| Missing vendor manifest | Complete section 2, including manifest generation; cloning alone is insufficient. |
| Modified upstream file | Restore the pinned checkout with unchanged file bytes. Do not edit the manifest to bypass a failed integrity check. |
| minepy C compilation error on Python 3.11 | Use the supplied installer, Cython 0.29.37, C++ Build Tools, and the project's pinned packages. |
| CUDA unavailable | Check `nvidia-smi`, the active environment, and the installed CUDA PyTorch wheel. |
| Missing real files | Correct the configured paths or disable the unavailable dataset explicitly. |
| Unset tolerance | Supply a finite nonnegative scalar or list before step 6. |
| Report identity mismatch | Rerun the relevant producing step and its dependents with consistent settings; do not edit saved identities manually. |
| CUDA out of memory | The trainer reduces physical batch size with accumulation. Close other GPU workloads; failure at batch size one needs further memory reduction or another device. |
| Windows file access error | Close programs holding output files, verify write permissions, and rerun the interrupted step. |
| Legacy SMPL pickle fails | Convert it in an already compatible licensed environment; do not rename an incompatible body model. |

Do not delete generated files from an active local run simply because they are not intended for GitHub. Keep them locally for resume and exclude them from version control.

## Third-party and data acknowledgments

This implementation uses selected IMUOptimize components through a thin adapter and the native minepy estimator. AMASS, SMPL, DIP-IMU, and TotalCapture must be obtained from their providers under their own terms. Please acknowledge the corresponding original publications when using them. The paper link and citation for this repository will be added when available.

# FieldGen <img src="icon.jpg" alt="icon" width="40">

<p align="center">
  <a href="https://fieldgen.github.io/">
    <img src="https://img.shields.io/badge/Project-Website-green" alt="Project Website">
  </a>
  <a href="https://arxiv.org/abs/2510.20774">
    <img src="https://img.shields.io/badge/arXiv-2510.20774-b31b1b" alt="arXiv">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License: MIT">
  </a>
  <a href="https://github.com/FieldGen/FieldGen">
    <img src="https://img.shields.io/github/last-commit/FieldGen/FieldGen" alt="GitHub last commit">
  </a>
  <a>
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
</p>


**[Website](https://fieldgen.github.io/) | [arXiv](https://arxiv.org/abs/2510.20774)**

Large-scale diverse data underpin robust manipulation, yet existing pipelines trade scale, diversity, and quality: simulation scales but leaves sim-to-real gaps; teleoperation is precise but costly and behaviorally narrow. FieldGen is a field-guided trajectory generation & data scaling method.

![teaser](./teaser.jpg)


Our method is very easy to use. No pipeline, no extra hardware and no heavy codebase. **Simply teleoperate the robot once to manipulate and move the arms around the objects, then run a single Python script—seamlessly transfer from traditional data collection.**

## ✨ Features
- Trajectory types: Bezier / Cone (structured approach path)
- Endpoint guided: local Z axis reverse assist in curve construction
- Non‑uniform sampling: denser near endpoint (stabler learning)
- Length alignment: unified by `chunk_size` (pad if short, truncate if long)
- Reward mode: `--reward` multiplies samples with multi‑radius endpoint perturbation + normalized distance reward
- Multi‑task batching: iterate multiple task folders
- Standardized HDF5 layout: action / state / eef / effector / reward
- Rich visualization: 3D scatter, multi‑view density HTML; trajectory + orientation arrows
- Centralized YAML configuration
- Minimal algorithms: easy to plug in new trajectory generators

## 📁 Repository Layout (key files)
```
fieldgen/
   generate.py              # main generation entry
   requirements.txt         # Python dependencies
   config/
      config.yaml           # main configuration
   utils/
      bezier_util.py        # Bezier curve utilities
      cone_util.py          # Cone half‑cycloid utilities
      rpy_util.py           # RPY / quaternion interpolation helpers
      visualize_points.py   # point & trajectory visualization
   scripts/                 # data conversion, splits, HDF5 inspection
   tests/                   # demo style tests
```

## 🚀 Installation
Python >= 3.10 recommended.
```bash
git clone <repo-url>
cd fieldgen
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Optional interactive visualization:
```bash
pip install plotly
```
Base dependencies (see `requirements.txt`): `numpy, scipy, pyyaml, h5py, Pillow, tqdm, matplotlib` (+ optional `plotly`).

## ⚙️ Configuration (`config/config.yaml`)
`generate` section (selected fields):

| Field | Meaning |
|-------|---------|
| curve_type | `bezier` or `cone` trajectory type |
| output_path | Root directory for generated episode outputs |
| chunk_size | Unified trajectory length T |
| beta | Secondary sampling ratio control (point count estimation) |
| endpoint_random_radius | Max perturbation radius in reward mode |
| multiplier | Augmentation factor per original point (reward mode) |
| reward_target_min / max | Clipping interval for normalized reward |

`tasks` section:
- key: task name (e.g. `pick0`)
- `path`: source directory (must contain `sample_points.h5`, optional `camera/*/*.jpg`)
- `max_trajectories`: limit number processed (null = all)

Example:
```yaml
generate:
   curve_type: cone
   output_path: /path/to/output
   chunk_size: 35
   beta: 0.0025
   endpoint_random_radius: 0.1
   multiplier: 10
   reward_target_min: 0.0
   reward_target_max: 1.0

tasks:
   pick0:
      path: /path/to/episode0
      max_trajectories: null
   pick1:
      path: /path/to/episode1
      max_trajectories: null
```

## 📥 Input Data Requirements
Each task folder must include:
```
<task_path>/
   sample_points.h5
   camera/
       0/*.jpg
       1/*.jpg   # subdirectories organized by eef index
```
Mandatory datasets inside `sample_points.h5`:
- `state/eef/position` : shape `(N, >=12)`; columns `[6:9]` starting xyz, `[9:12]` starting rpy
- `endpoint` : shape `(>=12,)`; `[6:9]` final xyz, `[9:12]` final rpy

## 📤 Output Format
Each generated `episodeK/aligned_joints.h5` contains:
```
timestamps : (T,)
action/eef/position : (T, 12)
state/eef/position  : (T, 12)
action/effector/position : (T, 2)   # normalized (÷90)
state/effector/position  : (T, 2)
reward/value : scalar or (T,)       # only in --reward mode
```
Trajectory concatenation ordering: `[left_xyz(3), left_rpy(3), right_xyz(3), right_rpy(3)] = 12`.
Images: All relevant `.jpg` copied to `episodeK/camera/0/`.

## 💡 Usage
Deterministic endpoint mode:
```bash
python generate.py --config config/config.yaml
```
Reward augmentation mode (random endpoints + reward writing):
```bash
python generate.py --config config/config.yaml --reward
```

## 📊 Data Collection

FieldGen supports two complementary data acquisition modes for building large, diverse approach (pre‑manipulation) datasets:

1. 🔄 **Automated Workspace Filling**: Use the helper script (see `document/data_collect_notice.md`) to auto‑calibrate a reachable workspace from a few existing episodes, then generate a dense set of coverage points via 3D Z‑order (Morton) or Hilbert space‑filling curves. Each point optionally receives a synthesized sinusoidal RPY perturbation sequence to enrich orientation diversity.
2. 🎮 **Teleoperation Traces**: Manually drive the robot to collect longer (≈1–2 min) free‑space approach trajectories while lightly rotating the wrist/gripper and keeping the target object centered in view. This yields high‑quality, contact‑proximal behavior diversity where automated methods may underperform.

Full details are documented here:
▶ [Data Collection Guide](document/data_collect_notice.md)


## 🎯 Reward Mechanism
In reward mode:
1. Read original endpoint `original_endpoint_pos`
2. Stratified sample perturbation distances within `[0, endpoint_random_radius * (1 - reward_target_min)]`
3. Sample random direction vectors and perturb endpoint
4. Normalize distance: `reward = 1 - dist / R_radius`, clip to `[reward_target_min, reward_target_max]`
5. Generate one trajectory per perturbed endpoint + associated `reward/value`

Extensible ideas: include orientation deviation, smoothness, obstacle clearance, etc.

## 🔄 Trajectory Algorithms
### 🎨 Quadratic Bezier Curve
- Control points: `start`, `end`, projection adjustment using `(end + direct)` direction
- Arc length estimate + power resampling (`power > 1` densifies tail; `<1` densifies head)
- Degenerate case (direct≈0): fall back to midpoint control

### 🔺 In-Cone Half‑Cycloid
- If start inside cone: direct half‑cycloid to apex
- If start outside cone: axial line inwards + half‑cycloid segment
- Allocate sample counts by relative estimated length of segments

### 🔄 RPY Interpolation
- Same‑hemisphere quaternion handling avoids long rotations
- Quadratic ease‑out for monotonically decreasing angular velocity
- Accumulate axis‑angle increments for sequence

## 📊 Visualization
```bash
python utils/visualize_points.py
```
Generates:
- `html/point_distribution_3d_<task>.html`
- `html/point_distribution_2d_density_<task>.html`
Open in your browser.

## ❓ FAQ
| Question | Suggestion |
|----------|------------|
| Trajectories too short (heavy padding) | Tune `beta` or reduce `chunk_size` |
| Not dense enough near endpoint | Increase internal Bezier/Cone power value |
| Reward variance too small | Increase `endpoint_random_radius` or lower `reward_target_min` |
| Add new trajectory type | Create `utils/xxx_curve.py` and extend factory in generator |
| Image copy slows processing | Limit per episode images or disable copying |

## 📄 License
This project is licensed under the MIT License – see the `LICENSE` file for details.

## 🤝 Contributing
PRs and issues are welcome:
- New trajectory generation algorithms & sampling strategies
- Data conversion & format enhancements
- Robust unit tests & CI integration

## 📚 Citation
If you find our work useful, please consider citing:
~~~
@article{wang2025fieldgen,
  title={FieldGen: From Teleoperated Pre-Manipulation Trajectories to Field-Guided Data Generation},
  author={Wang, Wenhao and Ye, Kehe and Zhou, Xinyu and Chen, Tianxing and Min, Cao and Zhu, Qiaoming and Yang, Xiaokang and Shen, Yongjian and Yang, Yang and Yao, Maoqing and others},
  journal={arXiv preprint arXiv:2510.20774},
  year={2025}
}
~~~

---
⭐ If FieldGen helps your research or project, consider starring the repository. Thank you! 🙏
---

# Automated Data Collection
`generate_points.py` — workspace point generation.
Automatically synthesize a broad coverage set of operational workspace points, then feed them sequentially to the robot for replay capture.

## Core Objective
Given the first-frame end-effector (EEF) poses from 3 (or more) collected episodes, infer workspace axis-aligned bounds and use a 3D Z-order (Morton) or Hilbert space-filling curve to generate up to `n` points inside that bounding box. For every point, synthesize a time-varying sinusoidal `rpy` sequence.

## Main Steps
1. Auto-calibration: Read `episodes_root/episode*/sample_points.h5` first frame EEF pose (indices 6:9 → xyz, 9:12 → rpy). Compute per-dimension min/max for xyz to obtain `point0/point1`. Use first-frame rpy (or averaged / user-specified) as initial orientation.
2. Compute minimal `order = k` such that `(2^k)^3 >= n`.
3. Decode Morton or Hilbert linear index → `(ix, iy, iz)` → normalize & map to physical space.
4. Build sinusoidal orientation: `rpy(t) = init_rpy + theta * sin(2π t / T)` with `t = i * dt`.
5. Write HDF5 datasets `/points/eef/(xyz|rpy|position)`; optional CSV export.

## Key Functions
| Function | Description |
|----------|-------------|
| `choose_order(n)` | Return minimal curve order bits |
| `morton_decode3` / `hilbert_decode3` | Decode linear index to integer grid coordinates |
| `generate_zorder_points` / `generate_hilbert_points` | Generate xyz point list |
| `synthesize_rpy_sequence` | Produce rpy sequence from sinusoid parameters |

## Output HDF5 Layout
```
/points/eef/xyz        (N,3)
/points/eef/rpy        (N,3)
/points/eef/position   (N,6)  # xyz + rpy concatenated
/meta/*                # generation metadata (point0, point1, thetas, periods, curve, dt, requested_n, generated_n ...)
```

## Common Arguments
| Arg | Meaning | Required | Default |
|-----|---------|----------|---------|
| `--n` | Desired number of points | yes | - |
| `--episodes-root` | Root directory for auto calibration (contains `episodeX` subfolders) | conditional* | - |
| `--episodes` | Explicit episode names used for calibration | no | first 3 episodes* |
| `--point0/--point1` | Manually specify workspace diagonal points (x,y,z) | no | auto-calibrated |
| `--init-rpy` | Initial rpy | no | first-frame rpy |
| `--theta` | Swing amplitudes (rad) `theta0,theta1,theta2` | no | 0,0,0 |
| `--period` | Swing periods (s) `T0,T1,T2` | no | 10,10,10 |
| `--dt` | Time step for rpy sampling | no | 0.05 |
| `--curve` | `zorder` or `hilbert` curve type | no | zorder |
| `--output` | HDF5 output path | no | generated_points.h5 |
| `--csv` | Optional CSV output path | no | None |

> If `point0/point1/init-rpy` are not provided you must supply `--episodes-root` (the script will read `episode*/sample_points.h5`).

## Examples
```bash
# Auto-calibrate + generate 5000 points (Z-order)
python generate_points.py \
  --episodes-root /root/workspace/processed.auto/separated \
  --episodes episode0 episode1 episode2 \
  --n 5000 --theta 0.1,0.05,0.0 --period 10,12,15 --dt 0.05 --output points.h5

# Manual workspace + Hilbert curve
python generate_points.py \
  --point0 0.3,0.1,0.6 --point1 0.8,0.4,1.0 \
  --init-rpy 0,0,0 --theta 0.1,0.05,0.0 --period 10,12,15 \
  --n 4096 --curve hilbert --output points_hilbert.h5
```

---

# Teleoperation Data Collection

If automatically generated points are insufficient or unsuitable, manual teleoperation remains a highly effective fallback / complementary strategy. It improves overall data quality and targeted coverage of challenging regions.

## Guidelines
1. Use the true task endpoint as the terminal state.
2. Record a spatial movement trajectory of about 1–2 minutes.
3. Apply slight continuous wrist/gripper rotations while moving.
4. Strive for uniform and broad coverage of the reachable workspace.
5. Keep the manipulated object visible and near the center of the camera frame.
6. Move more slowly when necessary to reduce motion blur in captured images.

---
These two collection modes (automated workspace fill + focused teleoperation) can be combined: first fill coarse approach diversity, then refine task-critical contact behaviors with expert teleop traces.
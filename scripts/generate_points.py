#!/usr/bin/env python3
"""
Auto-calibrate a workspace from the first frame of 3 collected episodes and generate a sequence of continuous points (xyz + rpy) inside the resulting axis-aligned box using a space-filling curve (Z-order / Hilbert).

Inputs:
1. point0 / point1: May be passed explicitly; if omitted, derive per-dimension min/max from first-frame EEF positions of selected episodes.
2. Initial rpy: May be passed explicitly; if omitted, uses average (or first) rpy from frames.
3. Sinusoidal rpy swing parameters: theta0,theta1,theta2 and periods T0,T1,T2 for roll,pitch,yaw oscillations.
4. Hyperparameter n: desired number of points (<= theoretical upper bound).

Implementation:
- Use 3D Z-order (Morton) or Hilbert traversal over a 2^k 3D grid; take first n points in traversal order.
- Choose minimal k so (2^k)^3 >= n.
- Map integer grid coordinates linearly into physical box [point0, point1].
- rpy(t_i) = rpy_init + theta * sin(2π * t_i / T), with t_i = i * dt.

Outputs:
- HDF5 file containing:
    /points/eef/position (N,6)
    /points/eef/xyz      (N,3)
    /points/eef/rpy      (N,3)
    /meta/*              configuration metadata.
- Optional CSV export.

Dependencies: standard library + numpy + h5py.

Examples:
python generate_points.py \
    --episodes-root /root/workspace/processed.spcpick.fieldgen/separated \
    --episodes episode0 episode1 episode2 \
    --n 5000 --theta 0.1,0.05,0.0 --period 10,12,15 \
    --dt 0.05 --output points.h5

Manual workspace specification:
python generate_points.py --point0 0.3,0.1,0.6 --point1 0.8,0.4,1.0 \
    --init-rpy 0,0,0 --theta 0.1,0.05,0.0 --period 10,12,15 --n 4096
"""
from __future__ import annotations
import argparse
import math
import pathlib
import sys
import numpy as np
import h5py
from typing import List, Tuple, Optional


def parse_vec3(arg: str, name: str) -> np.ndarray:
    try:
        parts = [float(x) for x in arg.split(',')]
        if len(parts) != 3:
            raise ValueError
        return np.array(parts, dtype=float)
    except Exception:
        raise argparse.ArgumentTypeError(f"{name} must be three comma-separated floats: x,y,z")


def parse_vec(arg: str, n: int, name: str) -> np.ndarray:
    try:
        parts = [float(x) for x in arg.split(',')]
        if len(parts) != n:
            raise ValueError
        return np.array(parts, dtype=float)
    except Exception:
        raise argparse.ArgumentTypeError(f"{name} must be {n} comma-separated floats")


def load_first_frames(episodes_root: pathlib.Path, episode_names: List[str]) -> List[np.ndarray]:
    frames = []
    for ep in episode_names:
        sample_path = episodes_root / ep / 'sample_points.h5'
        if not sample_path.exists():
            raise FileNotFoundError(f"Not found: {sample_path}")
        with h5py.File(sample_path, 'r') as f:
            # Compatible with state/eef/position or endpoint fallback
            if 'state' in f and 'eef' in f['state'] and 'position' in f['state']['eef']:
                data = np.array(f['state']['eef']['position'])
                first = data[0]
            elif 'endpoint' in f:
                first = np.array(f['endpoint'])
            else:
                raise KeyError(f"EEF position not found in {sample_path}")
            if first.shape[0] != 12:
                raise ValueError(f"First frame shape {first.shape} is not (12,) in {sample_path}")
            frames.append(first)
    return frames


def derive_workspace_and_rpy(frames: List[np.ndarray]):
    xyz = np.stack([f[6:9] for f in frames], axis=0)
    rpy = np.stack([f[9:12] for f in frames], axis=0)
    p0 = xyz.min(axis=0)
    p1 = xyz.max(axis=0)
    init_rpy = rpy[0]
    return p0, p1, init_rpy


def choose_order(n: int) -> int:
    # (2^k)^3 >= n  =>  k >= ceil(log2(n)/3)
    return max(1, math.ceil(math.log2(max(1, n)) / 3.0))


def morton_decode3(index: int, bits: int) -> Tuple[int, int, int]:
    """3D Morton (Z-order) decode.

    Args:
        index: linear Morton id
        bits: number of bits per axis (order)
    Returns:
        (x,y,z) integer coordinates in [0, 2^bits - 1]
    """
    x = y = z = 0
    for i in range(bits):
        x |= ((index >> (3 * i)) & 1) << i
        y |= ((index >> (3 * i + 1)) & 1) << i
        z |= ((index >> (3 * i + 2)) & 1) << i
    return x, y, z


def hilbert_decode3(index: int, bits: int) -> Tuple[int, int, int]:
    """3D Hilbert curve integer decode (Skilling transpose scheme).

    Reference: John Skilling, "Programming the Hilbert curve" (2004).
    Given Hilbert index (0 .. (2^(3*bits)) - 1), compute corresponding (x,y,z) grid coordinate.
    Orientation may differ from other libraries but preserves locality & space-filling properties.
    """
    n_dims = 3
    X = [0, 0, 0]
    # Expand Hilbert integer into transposed form (pre-bit-interleaving coordinate bits)
    for i in range(bits):
        for j in range(n_dims):
            # (n_dims -1 - j) match bit order
            bit = (index >> (i * n_dims + (n_dims - 1 - j))) & 1
            X[j] |= bit << i
    # Gray decode (adjacent points differ by one bit)
    t = X[-1] >> 1
    for i in range(n_dims - 2, -1, -1):
        X[i] ^= X[i + 1]
    X[0] ^= t
    # Iteratively derive true coordinates
    Q = 2
    while Q <= (1 << bits):
        P = Q - 1
        for i in range(n_dims - 1, -1, -1):
            if X[i] & Q:
                X[0] ^= P
            else:
                t = (X[0] ^ X[i]) & P
                X[0] ^= t
                X[i] ^= t
        Q <<= 1
    return X[0], X[1], X[2]


def generate_zorder_points(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    order = choose_order(n)
    size = 1 << order  # grid points per axis
    total = size ** 3
    m = min(n, total)
    scale = (p1 - p0)
    pts = np.zeros((m, 3), dtype=float)
    for i in range(m):
        x_i, y_i, z_i = morton_decode3(i, order)
    # Normalize to [0,1]
        fx = x_i / (size - 1) if size > 1 else 0.0
        fy = y_i / (size - 1) if size > 1 else 0.0
        fz = z_i / (size - 1) if size > 1 else 0.0
        pts[i] = p0 + scale * np.array([fx, fy, fz])
    return pts


def generate_hilbert_points(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    """Generate point sequence along 3D Hilbert curve.

    Same strategy as Z-order: choose minimal order s.t. (2^order)^3 >= n, take first n points.
    """
    order = choose_order(n)
    size = 1 << order
    total = size ** 3
    m = min(n, total)
    scale = (p1 - p0)
    pts = np.zeros((m, 3), dtype=float)
    for i in range(m):
        x_i, y_i, z_i = hilbert_decode3(i, order)
        fx = x_i / (size - 1) if size > 1 else 0.0
        fy = y_i / (size - 1) if size > 1 else 0.0
        fz = z_i / (size - 1) if size > 1 else 0.0
        pts[i] = p0 + scale * np.array([fx, fy, fz])
    return pts


def synthesize_rpy_sequence(init_rpy: np.ndarray, thetas: np.ndarray, periods: np.ndarray, N: int, dt: float) -> np.ndarray:
    rpy_seq = np.zeros((N, 3), dtype=float)
    for i in range(N):
        t = i * dt
        phase = 2 * math.pi * t / periods
        rpy_seq[i] = init_rpy + thetas * np.sin(phase)
    return rpy_seq


def save_h5(path: pathlib.Path, xyz: np.ndarray, rpy: np.ndarray, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        g_points = f.create_group('points')
        g_eef = g_points.create_group('eef')
        g_eef.create_dataset('xyz', data=xyz)
        g_eef.create_dataset('rpy', data=rpy)
        pos = np.concatenate([xyz, rpy], axis=1)
        g_eef.create_dataset('position', data=pos)
        g_meta = f.create_group('meta')
        for k, v in meta.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                g_meta.create_dataset(k, data=np.array(v))
            else:
                g_meta.attrs[k] = v


def save_csv(path: pathlib.Path, xyz: np.ndarray, rpy: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = 'x,y,z,roll,pitch,yaw'
    data = np.concatenate([xyz, rpy], axis=1)
    # Fixed decimal format, avoid scientific notation; keep 7 decimal places
    np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.7f')


def main():
    ap = argparse.ArgumentParser(description='Generate workspace point sequence via space-filling curve')
    ap.add_argument('--episodes-root', type=pathlib.Path, help='Root directory containing episode subfolders (for auto calibration)')
    ap.add_argument('--episodes', nargs='*', default=None, help='Episode names used for calibration (default: first 3 sorted)')
    ap.add_argument('--point0', type=str, default=None, help='Manually specify workspace corner 0: x,y,z')
    ap.add_argument('--point1', type=str, default=None, help='Manually specify workspace corner 1: x,y,z')
    ap.add_argument('--init-rpy', type=str, default=None, help='Initial rpy: r,p,y')
    ap.add_argument('--theta', type=str, default='0,0,0', help='Oscillation amplitudes theta0,theta1,theta2 (rad)')
    ap.add_argument('--period', type=str, default='10,10,10', help='Periods T0,T1,T2 (seconds)')
    ap.add_argument('--n', type=int, required=True, help='Desired number of points')
    ap.add_argument('--dt', type=float, default=0.05, help='Time step dt (seconds) for rpy sampling')
    ap.add_argument('--curve', type=str, choices=['zorder', 'hilbert'], default='zorder', help='Space-filling curve type')
    ap.add_argument('--output', type=pathlib.Path, default=pathlib.Path('generated_points.h5'), help='Output HDF5 path')
    ap.add_argument('--csv', type=pathlib.Path, default=None, help='Optional CSV output path')
    args = ap.parse_args()

    # Parse swing parameters
    thetas = parse_vec(args.theta, 3, 'theta')
    periods = parse_vec(args.period, 3, 'period')
    if np.any(periods <= 0):
        ap.error('Each period component must be > 0')

    # Workspace & initial rpy
    if (args.point0 is None or args.point1 is None or args.init_rpy is None):
        if args.episodes_root is None:
            ap.error('point0/point1/init-rpy not provided; must supply --episodes-root for auto calibration')
        ep_root = args.episodes_root
        if not ep_root.exists():
            ap.error(f'episodes-root {ep_root} does not exist')
        if args.episodes is None or len(args.episodes) == 0:
            # Auto-pick first 3 episode* directories
            all_eps = sorted([p.name for p in ep_root.iterdir() if p.is_dir() and p.name.startswith('episode')])
            if len(all_eps) < 1:
                ap.error('No episode* directories found under episodes-root')
            chosen = all_eps[:3]
        else:
            chosen = args.episodes
        frames = load_first_frames(ep_root, chosen)
        p0_auto, p1_auto, init_rpy_auto = derive_workspace_and_rpy(frames)
        if args.point0 is None:
            p0 = p0_auto
        else:
            p0 = parse_vec3(args.point0, 'point0')
        if args.point1 is None:
            p1 = p1_auto
        else:
            p1 = parse_vec3(args.point1, 'point1')
        if args.init_rpy is None:
            init_rpy = init_rpy_auto
        else:
            init_rpy = parse_vec(args.init_rpy, 3, 'init-rpy')
    else:
        p0 = parse_vec3(args.point0, 'point0')
        p1 = parse_vec3(args.point1, 'point1')
        init_rpy = parse_vec(args.init_rpy, 3, 'init-rpy')

    # Normalize: ensure p0 is the minimum corner
    lo = np.minimum(p0, p1)
    hi = np.maximum(p0, p1)
    p0, p1 = lo, hi

    # Generate spatial sequence
    if args.curve == 'zorder':
        xyz = generate_zorder_points(p0, p1, args.n)
    elif args.curve == 'hilbert':
        xyz = generate_hilbert_points(p0, p1, args.n)
    else:
        raise NotImplementedError(args.curve)

    # Generate rpy sequence
    rpy = synthesize_rpy_sequence(init_rpy, thetas, periods, xyz.shape[0], args.dt)

    meta = {
        'point0': p0,
        'point1': p1,
        'init_rpy': init_rpy,
        'thetas': thetas,
        'periods': periods,
        'dt': args.dt,
        'curve': args.curve,
        'requested_n': args.n,
        'generated_n': xyz.shape[0],
    }

    save_h5(args.output, xyz, rpy, meta)
    if args.csv is not None:
        save_csv(args.csv, xyz, rpy)

    print(f'[OK] Generated {xyz.shape[0]} points -> {args.output}')
    if args.csv:
        print(f'[OK] CSV written: {args.csv}')
    print(f'Workspace corners: point0={p0}, point1={p1}')
    print(f'Initial rpy: {init_rpy}, theta={thetas}, period={periods}')


if __name__ == '__main__':
    main()

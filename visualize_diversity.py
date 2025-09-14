#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化点的多样性 (voxel 覆盖情况)

功能概述：
1. 读取给定路径 (目录或直接 h5 文件)，加载其中的 sample_points.h5（或指定文件）。
2. 从数据集中抽取末端执行器的位置点 (state/eef/position) 的第 7~9 列 (索引 6:9) 作为 (x,y,z)。
3. 计算所有点的最小/最大 xyz，构成包围盒。
4. 使用 beta 定义体素尺寸：voxel_size = beta * max(range_xyz)。beta 越小 -> 体素越细。
5. 对包围盒做均匀划分，生成每个体素的中心点；判断哪些体素被点“覆盖”。
   覆盖判定（两种方法，可选）：
   - bin（默认）: 直接把点落入所在的 voxel 索引，标记占用。
   - nn : 为每个 voxel center 找最近点，若该点位于该 voxel 边界内则占用（更慢）。
6. 生成 Plotly 3D 可视化（颜色可配置）：
    - 占用体素：默认橙色 (#ff7f0e)
    - 空体素：默认半透明灰 (rgba(120,120,120,0.15))（可抽样显示，避免过多点导致浏览器卡顿）
7. 输出统计信息与 HTML。
8. 新增“直接点”模式：使用 --direct-points 输出所有收集到的原始点（不做体素化），可通过 --direct-max-points 控制最大点数（随机下采样）。

使用示例：
python visualize_diversity.py --path /data/episode0 \
    --beta 0.05 --method bin --max-total-voxels 300000 --sample-empty 40000

参数说明（新版支持 episode* 目录批量聚合）：
--path               可以是：
                     1) 单个 H5 文件路径 (aligned_joints.h5)
                     2) 单个 episode 目录（包含 aligned_joints.h5）
                     3) 训练根目录（包含多个 episode*/ 子目录）
--h5-name            当遍历目录时要寻找的文件名（默认 aligned_joints.h5）。
--dataset            位置数据集路径；若未提供或未找到将自动在文件内智能搜索 (列>=9)。
--episodes           只加载指定 episode；逗号分隔或范围 0-99，可混合: 0-9,15,23。
--per-episode-max    每个 episode 采样的最大点数 (0=不限制)，可随机下采样。
--global-max-points  全局最大点数上限，超出则再随机下采样 (0=不限制)。
--beta               控制体素尺寸 (voxel_size = beta * max_range)。
--method             占用计算：bin 或 nn。
--max-total-voxels   体素数量安全上限，超过自动放大体素。
--sample-empty       可视化时空体素采样数 (0=不显示)。
--seed               随机种子。
--output             输出 HTML (默认 html/diversity_voxels.html)。
"""

import os
import argparse
import h5py
import numpy as np
import plotly.graph_objects as go
from math import ceil
from typing import List, Tuple, Optional

try:
    from scipy.spatial import cKDTree  # 仅在 method==nn 时需要
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _auto_find_dataset(f: h5py.File, min_cols: int = 9) -> Optional[str]:
    """自动搜索第一批 shape[1] >= min_cols 的二维数据集。
    优先匹配常见关键词顺序: state/eef/position -> position -> eef -> state。
    """
    candidates = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            shape = obj.shape
            if len(shape) == 2 and shape[1] >= min_cols:
                candidates.append(name)
    f.visititems(visit)
    if not candidates:
        return None
    priority = ['state/eef/position', 'position', 'eef', 'state']
    for p in priority:
        for c in candidates:
            if c.endswith(p):
                return c
    # fallback 取最短路径（更可能是核心）
    candidates.sort(key=lambda x: len(x))
    return candidates[0]


def load_points(h5_path: str, dataset: Optional[str] = None) -> Tuple[np.ndarray, str]:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"未找到 H5 文件: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        ds_to_use = dataset
        if ds_to_use and ds_to_use not in f:
            print(f"警告：指定数据集 '{ds_to_use}' 不存在，将尝试自动搜索。")
            ds_to_use = None
        if not ds_to_use:
            ds_to_use = _auto_find_dataset(f)
            if not ds_to_use:
                raise KeyError(f"文件中未找到满足列数>=9 的二维数据集: {h5_path}")
        data = np.asarray(f[ds_to_use])
    if data.shape[1] < 9:
        raise ValueError(f"数据集 '{ds_to_use}' 列数不足 9 (实际 {data.shape[1]})，无法提取 6:9 作为 xyz")
    points_xyz = data[:, 6:9]
    return points_xyz, ds_to_use


def parse_episode_filter(ep_str: str) -> List[int]:
    result = []
    for part in ep_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            if a.isdigit() and b.isdigit():
                start, end = int(a), int(b)
                if start > end:
                    start, end = end, start
                result.extend(range(start, end + 1))
        elif part.isdigit():
            result.append(int(part))
    # 去重排序
    return sorted(set(result))


def collect_points(path: str, h5_name: str, dataset: Optional[str], episodes_filter: Optional[List[int]], per_episode_max: int, global_max: int) -> Tuple[np.ndarray, str, List[Tuple[int, int]]]:
    """收集 path 下所有 episode* 目录中的点。
    返回 (points, 使用的数据集名称, 统计列表[(ep_id, 点数), ...])"""
    stats = []
    collected = []
    used_dataset = None

    def _load_one(h5_file: str, ep_id: int):
        nonlocal used_dataset
        pts, dsname = load_points(h5_file, dataset)
        if used_dataset is None:
            used_dataset = dsname
        elif used_dataset != dsname:
            # 不同数据集名称但结构兼容，我们仍然使用
            pass
        if per_episode_max > 0 and len(pts) > per_episode_max:
            idx = np.random.choice(len(pts), per_episode_max, replace=False)
            pts = pts[idx]
        stats.append((ep_id, len(pts)))
        collected.append(pts)

    if os.path.isfile(path):
        print("检测到传入的是单个 H5 文件。")
        _load_one(path, ep_id=-1)
    elif os.path.isdir(path):
        # 判断是否是单个 episode 目录
        candidate_file = os.path.join(path, h5_name)
        if os.path.isfile(candidate_file):
            ep_id = _extract_episode_id(path)
            print(f"检测到单个 episode 目录: {path} (episode{ep_id})")
            _load_one(candidate_file, ep_id=ep_id)
        else:
            # 视为根目录，遍历 episode*
            subs = sorted([d for d in os.listdir(path) if d.startswith('episode')])
            if episodes_filter is not None:
                subs = [d for d in subs if _extract_episode_id(os.path.join(path, d)) in episodes_filter]
            if not subs:
                raise RuntimeError("未找到任何 episode* 子目录。")
            print(f"发现 {len(subs)} 个 episode 子目录，开始加载……")
            for d in subs:
                ep_dir = os.path.join(path, d)
                h5_file = os.path.join(ep_dir, h5_name)
                if not os.path.isfile(h5_file):
                    print(f"跳过 (无 {h5_name}): {d}")
                    continue
                ep_id = _extract_episode_id(d)
                if (episodes_filter is not None) and (ep_id not in episodes_filter):
                    continue
                try:
                    _load_one(h5_file, ep_id)
                except Exception as e:
                    print(f"加载 {d} 失败: {e}")
    else:
        raise FileNotFoundError(f"路径不存在: {path}")

    if not collected:
        raise RuntimeError("未收集到任何点。")
    all_points = np.concatenate(collected, axis=0)
    if global_max > 0 and len(all_points) > global_max:
        idx = np.random.choice(len(all_points), global_max, replace=False)
        all_points = all_points[idx]
    return all_points, (used_dataset or (dataset or 'UNKNOWN')), stats


def _extract_episode_id(name: str) -> int:
    # name 可能是 "episode123" 或完整路径
    base = os.path.basename(name.rstrip('/'))
    if base.startswith('episode'):
        suf = base[len('episode'):]
        if suf.isdigit():
            return int(suf)
    return -1


def compute_voxel_grid(points: np.ndarray, beta: float, max_total_voxels: int):
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    ranges = max_xyz - min_xyz
    max_range = float(ranges.max())
    if max_range <= 0:
        raise ValueError("所有点重合，无法构建体素网格。")

    # 初始 voxel 尺寸
    voxel_size = beta * max_range
    if voxel_size <= 0:
        raise ValueError("beta 太小或范围异常，voxel_size <= 0")

    def _calc_counts(vsize):
        counts = np.maximum(1, np.ceil(ranges / vsize).astype(int))
        return counts, int(np.prod(counts))

    counts, total = _calc_counts(voxel_size)
    # 自适应放大，防止体素过多
    if total > max_total_voxels:
        scale = (total / max_total_voxels) ** (1 / 3)
        voxel_size *= scale * 1.05  # 放大一点点，留冗余
        counts, total = _calc_counts(voxel_size)

    nx, ny, nz = counts.tolist()

    # 构建中心坐标
    def _centers(min_v, n, vsize):
        # 生成 n 个体素，其中心在： min_v + (0.5 + i)*vsize
        return min_v + (np.arange(n) + 0.5) * (ranges[np.argmax(ranges)]/ranges[np.argmax(ranges)] * vsize if vsize>0 else vsize) if False else min_v + (np.arange(n) + 0.5) * voxel_size

    xs = _centers(min_xyz[0], nx, voxel_size)
    ys = _centers(min_xyz[1], ny, voxel_size)
    zs = _centers(min_xyz[2], nz, voxel_size)

    # 实际最大覆盖到的坐标（用于 NN 判断边界）
    grid_info = {
        'min': min_xyz,
        'max': max_xyz,
        'ranges': ranges,
        'voxel_size': voxel_size,
        'counts': (nx, ny, nz),
        'total_voxels': total,
        'xs': xs,
        'ys': ys,
        'zs': zs,
    }
    return grid_info


def occupancy_by_binning(points: np.ndarray, grid_info):
    min_xyz = grid_info['min']
    voxel_size = grid_info['voxel_size']
    nx, ny, nz = grid_info['counts']

    # 计算索引（落在边界右侧的点 clip 到最后一个体素）
    idx = np.floor((points - min_xyz) / voxel_size).astype(int)
    idx = np.clip(idx, 0, np.array([nx - 1, ny - 1, nz - 1]))

    occ = np.zeros((nx, ny, nz), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def occupancy_by_nn(points: np.ndarray, grid_info):  # pragma: no cover (性能向，测试可跳)
    if not _HAS_SCIPY:
        raise RuntimeError("需要 scipy.spatial.cKDTree 支持 nn 方法，请先安装 scipy 或改用 bin 方法。")
    voxel_size = grid_info['voxel_size']
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    nx, ny, nz = grid_info['counts']
    centers = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)

    tree = cKDTree(points)
    dists, idxs = tree.query(centers, k=1)

    # 判断最近点是否在该体素立方体中（按中心 ± voxel_size/2 的轴对齐盒）
    nearest_pts = points[idxs]
    inside = np.all(np.abs(nearest_pts - centers) <= (voxel_size / 2.0), axis=1)

    occ = inside.reshape(nx, ny, nz)
    return occ


def build_plot(occ, grid_info, sample_empty: int, output: str, occupied_color: str, empty_color: str):
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    nx, ny, nz = grid_info['counts']
    voxel_size = grid_info['voxel_size']

    # 体素中心坐标展开
    Xc, Yc, Zc = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1)
    occ_flat = occ.ravel()

    occupied_centers = centers[occ_flat]
    empty_centers = centers[~occ_flat]

    traces = []
    if len(occupied_centers) > 0:
        traces.append(
            go.Scatter3d(
                x=occupied_centers[:, 0],
                y=occupied_centers[:, 1],
                z=occupied_centers[:, 2],
                mode='markers',
                name='占用体素',
                marker=dict(size=3, color=occupied_color, opacity=0.9)
            )
        )
    if sample_empty > 0 and len(empty_centers) > 0:
        if len(empty_centers) > sample_empty:
            sel_idx = np.random.choice(len(empty_centers), sample_empty, replace=False)
            empty_show = empty_centers[sel_idx]
        else:
            empty_show = empty_centers
        traces.append(
            go.Scatter3d(
                x=empty_show[:, 0],
                y=empty_show[:, 1],
                z=empty_show[:, 2],
                mode='markers',
                name='空体素(采样)',
                marker=dict(size=2, color=empty_color)
            )
        )

    occ_ratio = (len(occupied_centers) / (nx * ny * nz)) if (nx * ny * nz) else 0.0

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(
            f"Voxel 覆盖可视化 - 占用体素 {len(occupied_centers)}/总数 {nx*ny*nz} "
            f"(占用率 {occ_ratio*100:.2f}%)<br>voxel_size={voxel_size:.4g}, grid=({nx},{ny},{nz})"
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.write_html(output)
    return output, len(occupied_centers), nx * ny * nz, occ_ratio


def parse_args():
    p = argparse.ArgumentParser(description='点多样性体素覆盖可视化 / 直接点查看 (支持批量 episode)')
    p.add_argument('--path', required=True, help='根目录 / episode 目录 / 单个 h5 文件')
    p.add_argument('--h5-name', default='aligned_joints.h5', help='遍历目录时寻找的文件名')
    p.add_argument('--dataset', default=None, help='数据集名称 (留空=自动检测)')
    p.add_argument('--episodes', default=None, help='只加载指定 episodes: 0-10,15,23')
    p.add_argument('--per-episode-max', type=int, default=0, help='每个 episode 最大采样点数 (0=不限制)')
    p.add_argument('--global-max-points', type=int, default=0, help='全局最大点数 (0=不限制)')
    p.add_argument('--beta', type=float, default=0.005, help='voxel_size = beta * max_range')
    p.add_argument('--method', choices=['bin', 'nn'], default='bin', help='占用计算方法')
    p.add_argument('--max-total-voxels', type=int, default=100000, help='体素数量安全上限')
    p.add_argument('--sample-empty', type=int, default=40000, help='可视化时空体素最大抽样数 (0=不显示)')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    p.add_argument('--occupied-color', default='#ff7f0e', help='占用体素颜色 (默认橙色 #ff7f0e)')
    p.add_argument('--empty-color', default='rgba(120,120,120,0.15)', help='空体素颜色 (默认半透明灰)')
    p.add_argument('--output', default='html/diversity_voxels.html', help='输出 HTML 文件路径')
    # 新增：直接点输出模式
    p.add_argument('--direct-points', action='store_true', help='启用直接点模式：跳过体素，输出所有点的3D散点')
    p.add_argument('--direct-max-points', type=int, default=0, help='直接点模式下最大点数 (0=不限制，超过将随机下采样)')
    p.add_argument('--direct-color', default='Viridis', help='直接点模式颜色映射 (Plotly colorscale 或单色)')
    p.add_argument('--direct-size', type=float, default=2, help='直接点模式单点尺寸')
    p.add_argument('--direct-output', default='html/diversity_points.html', help='直接点模式输出 HTML')
    return p.parse_args()


def build_direct_points_plot(points: np.ndarray, output: str, colorscale: str, size: float, single_color: Optional[str] = None):
    os.makedirs(os.path.dirname(output), exist_ok=True)

    def _hex_to_rgb(h: str):
        h = h.lstrip('#')
        if len(h) == 3:
            h = ''.join(c*2 for c in h)
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    if single_color:
        # 生成基于 single_color 的渐变 colorscale: 亮色 -> 原色
        try:
            r, g, b = _hex_to_rgb(single_color)
            # 生成一个更亮的起始色（与白色混合）
            mix_factor = 0.75  # 越大越接近白
            r_l = int(r + (255 - r) * mix_factor)
            g_l = int(g + (255 - g) * mix_factor)
            b_l = int(b + (255 - b) * mix_factor)
            custom_scale = [
                [0.0, f'rgb({r_l},{g_l},{b_l})'],
                [1.0, f'rgb({r},{g},{b})']
            ]
            marker = dict(
                size=size,
                color=points[:, 2],
                colorscale=custom_scale,
                opacity=0.8,
                colorbar=dict(title='Z')
            )
            title_mode = f'基于 {single_color} 渐变'
        except Exception:
            # 回退为纯单色（解析失败）
            marker = dict(size=size, color=single_color, opacity=0.8)
            title_mode = f'单色 {single_color}'
    else:
        marker = dict(
            size=size,
            color=points[:, 2],
            colorscale=colorscale if colorscale else 'Viridis',
            opacity=0.8,
            colorbar=dict(title='Z')
        )
        title_mode = f'colorscale={colorscale if colorscale else "Viridis"}'

    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=marker,
            name=f'points ({len(points)})'
        )
    ])
    fig.update_layout(
        title=f'直接点可视化 (共 {len(points)} 个点) - {title_mode}',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(r=20, b=10, l=10, t=40)
    )
    fig.write_html(output)
    return output


def main():
    args = parse_args()
    np.random.seed(args.seed)

    episodes_filter = None
    if args.episodes:
        episodes_filter = parse_episode_filter(args.episodes)
        print(f"过滤 episode 列表: {episodes_filter}")

    print("开始收集点 ...")
    points, dsname, stats = collect_points(
        path=args.path,
        h5_name=args.h5_name,
        dataset=args.dataset,
        episodes_filter=episodes_filter,
        per_episode_max=args.per_episode_max,
        global_max=args.global_max_points,
    )
    print(f"使用数据集: {dsname}")
    total_loaded = sum(s for _, s in stats)
    print(f"汇总得到 {len(points)} 个点(采样后)，原始累计 {total_loaded}。 Episode 分布示例: {stats[:10]}{' ...' if len(stats)>10 else ''}")

    # 直接点模式
    if args.direct_points:
        if args.direct_max_points > 0 and len(points) > args.direct_max_points:
            sel = np.random.choice(len(points), args.direct_max_points, replace=False)
            points_vis = points[sel]
            print(f"直接点模式：随机下采样 {len(points)} -> {len(points_vis)}")
        else:
            points_vis = points
        # 使用 occupied-color 作为单色。如果用户仍想用 colorscale，可以传入 --direct-color 'auto'
        use_single = True
        single_color = args.occupied_color if use_single else None
        # 如果用户显式指定 direct-color 且不是 'auto'，则按照 colorscale/单色判定：如果以#或rgb开头则覆盖 single_color
        if args.direct_color and args.direct_color.lower() != 'auto':
            if args.direct_color.lower().startswith(('#', 'rgb')):
                single_color = args.direct_color  # 用户自定义单色
            else:
                single_color = None  # 使用提供的 colorscale
                colorscale = args.direct_color
        else:
            colorscale = 'Viridis'  # 默认 colorscale 仅在 single_color 关闭时使用
        out_file = build_direct_points_plot(points_vis, args.direct_output, colorscale=args.direct_color, size=args.direct_size, single_color=single_color)
        print("\n================ 直接点模式完成 ================")
        print(f"输出文件: {os.path.abspath(out_file)}")
        print(f"显示点数: {len(points_vis)} (原始 {len(points)})")
        if single_color:
            print(f"渲染模式: 单色 {single_color}")
        else:
            print(f"渲染模式: colorscale {colorscale}")
        print("==============================================\n")
        return
    print("计算包围盒与体素网格……")

    grid_info = compute_voxel_grid(points, beta=args.beta, max_total_voxels=args.max_total_voxels)
    nx, ny, nz = grid_info['counts']
    print(f"体素网格大小: {nx} x {ny} x {nz} = {grid_info['total_voxels']} (voxel_size={grid_info['voxel_size']:.6f})")

    # 占用计算
    if args.method == 'bin':
        occ = occupancy_by_binning(points, grid_info)
    else:
        print("使用 NN 方法（可能较慢）…")
        occ = occupancy_by_nn(points, grid_info)

    # 构建并保存可视化
    output, occ_count, total_voxels, occ_ratio = build_plot(
        occ,
        grid_info,
        args.sample_empty,
        args.output,
        args.occupied_color,
        args.empty_color,
    )

    print("\n================ 统计信息 ================")
    print(f"总体素数: {total_voxels}")
    print(f"占用体素: {occ_count}")
    print(f"占用率: {occ_ratio*100:.2f}%")
    print(f"输出文件: {os.path.abspath(output)}")
    print("==========================================\n")


if __name__ == '__main__':
    main()

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
6. 生成 Plotly 3D 可视化：
   - 绿色：被占用的 voxel 中心
   - 半透明灰色：未占用的 voxel（可抽样显示，避免过多点导致浏览器卡顿）
7. 输出统计信息与 HTML。

使用示例：
python visualize_diversity.py --path /data/episode0 \
    --beta 0.05 --method bin --max-total-voxels 300000 --sample-empty 40000

参数说明（新版支持 episode* 目录批量聚合 & 多 path 优先级 / 可选多数据集）：
--path               可以是：
                     1) 单个 H5 文件路径 (aligned_joints.h5)
                     2) 单个 episode 目录（包含 aligned_joints.h5）
                     3) 训练根目录（包含多个 episode*/ 子目录）
                     可多次提供：--path A --path B --path C，按给定顺序设定优先级(浅->深)。
--h5-name            当遍历目录时要寻找的文件名（默认 aligned_joints.h5）。
--dataset            单个数据集名称 (与 --datasets 互斥)；若未提供且未指定 --datasets，将自动搜索一个 (列>=9)。
--datasets           多个数据集名称，逗号分隔。例如: state/eef/position,other/pos
                     顺序代表优先级 (前>后)，并用于颜色由浅到深。
--episodes           只加载指定 episode；逗号分隔或范围 0-99，可混合: 0-9,15,23。
--per-episode-max    每个 episode 采样的最大点数 (0=不限制)，可随机下采样。
--global-max-points  全局最大点数上限，超出则再随机下采样 (0=不限制)。
--beta               控制体素尺寸 (voxel_size = beta * max_range)。
--method             占用计算：bin 或 nn。
--max-total-voxels   体素数量安全上限，超过自动放大体素。
--sample-empty       可视化时空体素采样数 (0=不显示)。
--seed               随机种子。
--output             输出 HTML (默认 html/diversity_voxels.html)。
--color-mode         颜色模式：
                     green=统一绿色(当前默认) / gradient=按来源梯度绿 / category=按来源分类色
--colors             自定义分类颜色列表(逗号分隔)，与 category 模式配合；数量不足会循环。
"""

import os
import argparse
import h5py
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import string
from math import ceil
from typing import List, Tuple, Optional, Dict
# 全局字体配置常量
LABEL_FONT_SIZE = 14  # 坐标轴字母 (X/Y/Z) 以及 Matplotlib 轴标签
LEGEND_FONT_SIZE = 12  # 图例字体
AXIS_LETTER_FONT_SIZE = 20  # Plotly 自绘轴字母大小
# 多样性等级标签 (固定三层)
CATEGORY_DIVERSITY_LABELS = ['Diversity High', 'Diversity Middle', 'Diversity Low']
LEGEND_MARKER_SCALE = 3.0
LEGEND_MARKER_MIN = 10

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


def collect_points_single_dataset(path: str, h5_name: str, dataset: Optional[str], episodes_filter: Optional[List[int]], per_episode_max: int, global_max: int) -> Tuple[np.ndarray, str, List[Tuple[int, int]]]:
    """旧逻辑：单数据集收集。"""
    stats: List[Tuple[int, int]] = []
    collected: List[np.ndarray] = []
    used_dataset: Optional[str] = None

    def _load_one(h5_file: str, ep_id: int):
        nonlocal used_dataset
        pts, dsname = load_points(h5_file, dataset)
        if used_dataset is None:
            used_dataset = dsname
        if per_episode_max > 0 and len(pts) > per_episode_max:
            idx = np.random.choice(len(pts), per_episode_max, replace=False)
            pts = pts[idx]
        stats.append((ep_id, len(pts)))
        collected.append(pts)

    _iterate_files(path, h5_name, episodes_filter, _load_one)

    if not collected:
        raise RuntimeError("未收集到任何点。")
    all_points = np.concatenate(collected, axis=0)
    if global_max > 0 and len(all_points) > global_max:
        idx = np.random.choice(len(all_points), global_max, replace=False)
        all_points = all_points[idx]
    return all_points, (used_dataset or (dataset or 'UNKNOWN')), stats


def _iterate_files(path: str, h5_name: str, episodes_filter: Optional[List[int]], cb):
    """遍历 path，针对每个发现的 h5 文件调用回调 cb(h5_file, ep_id)。"""
    if os.path.isfile(path):
        print("检测到传入的是单个 H5 文件。")
        cb(path, -1)
    elif os.path.isdir(path):
        candidate_file = os.path.join(path, h5_name)
        if os.path.isfile(candidate_file):
            ep_id = _extract_episode_id(path)
            print(f"检测到单个 episode 目录: {path} (episode{ep_id})")
            cb(candidate_file, ep_id)
        else:
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
                    continue
                ep_id = _extract_episode_id(d)
                if (episodes_filter is not None) and (ep_id not in episodes_filter):
                    continue
                try:
                    cb(h5_file, ep_id)
                except Exception as e:
                    print(f"加载 {d} 失败: {e}")
    else:
        raise FileNotFoundError(f"路径不存在: {path}")


def collect_points_multi_datasets(path: str, h5_name: str, datasets: List[str], episodes_filter: Optional[List[int]], per_episode_max: int, global_max: int) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Tuple[int, int]]]]:
    """多数据集收集：返回 {dataset: points} 及 {dataset: stats_list[(ep_id,count), ...]}"""
    collected: Dict[str, List[np.ndarray]] = {d: [] for d in datasets}
    stats: Dict[str, List[Tuple[int, int]]] = {d: [] for d in datasets}

    def _load_one(h5_file: str, ep_id: int):
        with h5py.File(h5_file, 'r') as f:
            for ds in datasets:
                if ds not in f:
                    # 静默跳过缺失数据集
                    continue
                arr = np.asarray(f[ds])
                if arr.ndim != 2 or arr.shape[1] < 9:
                    continue
                pts = arr[:, 6:9]
                if per_episode_max > 0 and len(pts) > per_episode_max:
                    idx = np.random.choice(len(pts), per_episode_max, replace=False)
                    pts = pts[idx]
                collected[ds].append(pts)
                stats[ds].append((ep_id, len(pts)))

    _iterate_files(path, h5_name, episodes_filter, _load_one)

    result_points: Dict[str, np.ndarray] = {}
    for ds in datasets:
        if not collected[ds]:
            print(f"数据集 {ds} 未收集到任何点。")
            continue
        merged = np.concatenate(collected[ds], axis=0)
        if global_max > 0 and len(merged) > global_max:
            idx = np.random.choice(len(merged), global_max, replace=False)
            merged = merged[idx]
        result_points[ds] = merged
    if not result_points:
        raise RuntimeError("所有指定数据集均未收集到点。")
    return result_points, stats


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


def build_plot(occ, grid_info, sample_empty: int, output: str):
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    nx, ny, nz = grid_info['counts']
    voxel_size = grid_info['voxel_size']

    # 体素中心坐标展开
    Xc, Yc, Zc = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1)
    occ_flat = occ.ravel()

    occupied_centers = centers[occ_flat]
    empty_centers = centers[~occ_flat]

    traces = [go.Scatter3d(
        x=occupied_centers[:, 0],
        y=occupied_centers[:, 1],
        z=occupied_centers[:, 2],
        mode='markers',
        name='占用体素',
        marker=dict(size=3, color='green', opacity=0.9)
    )]

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
    p = argparse.ArgumentParser(description='点多样性体素覆盖可视化 (支持批量 episode + 多 path 优先级 + 多数据集)')
    p.add_argument('--path', action='append', required=True, help='根目录 / episode 目录 / 单个 h5 文件；可多次指定以形成优先级顺序')
    p.add_argument('--h5-name', default='aligned_joints.h5', help='遍历目录时寻找的文件名')
    g_ds = p.add_mutually_exclusive_group()
    g_ds.add_argument('--dataset', default=None, help='单数据集名称 (留空=自动检测一个)')
    g_ds.add_argument('--datasets', default=None, help='多数据集名称, 逗号分隔，顺序=优先级(浅->深)')
    p.add_argument('--episodes', default=None, help='只加载指定 episodes: 0-10,15,23')
    p.add_argument('--per-episode-max', type=int, default=0, help='每个 episode 最大采样点数 (0=不限制)')
    p.add_argument('--global-max-points', type=int, default=0, help='全局最大点数 (0=不限制) 对单/多数据集分别应用')
    p.add_argument('--beta', type=float, default=0.05, help='voxel_size = beta * max_range')
    p.add_argument('--method', choices=['bin', 'nn'], default='bin', help='占用计算方法')
    p.add_argument('--max-total-voxels', type=int, default=300000, help='体素数量安全上限')
    p.add_argument('--sample-empty', type=int, default=40000, help='可视化时空体素最大抽样数 (0=不显示)')
    p.add_argument('--seed', type=int, default=0, help='随机种子')
    p.add_argument('--output', default='html/diversity_voxels.html', help='输出 HTML 文件路径')
    p.add_argument('--color-mode', choices=['green', 'gradient', 'category'], default='green', help='颜色模式：green|gradient|category')
    p.add_argument('--colors', default=None, help='自定义分类颜色列表(逗号分隔)，用于 category 模式')
    p.add_argument('--display-mode', choices=['voxel','raw'], default='voxel', help='展示模式：voxel=体素 raw=直接点云')
    p.add_argument('--raw-max-points', type=int, default=200000, help='raw 模式最大显示点数')
    p.add_argument('--raw-2d-plane', choices=['xy','xz','yz'], default='xy', help='raw 模式 2D 投影平面')
    p.add_argument('--raw-2d-output', default=None, help='raw 模式 2D 图片输出路径 (默认=与 HTML 同名加 _2d.png)')
    p.add_argument('--raw-3d-output', default=None, help='raw 模式 3D 视角投影图片输出 (PNG，默认=与 HTML 同名加 _3d.png)')
    p.add_argument('--raw-3d-elev', type=float, default=20.0, help='3D 视角仰角 (deg)')
    p.add_argument('--raw-3d-azim', type=float, default=-60.0, help='3D 视角方位角 (deg)')
    p.add_argument('--raw-3d-size', type=float, nargs=2, default=(6,6), help='3D 视角图片尺寸 (inch 宽 高)')
    p.add_argument('--raw-individual-dir', default=None, help='RAW 多 path 时：为每个 path 生成单独拆分图 (HTML + 2D + 3D) 的输出目录')
    p.add_argument('--alpha', type=float, default=0.8, help='点/体素不透明度 (0-1)')
    p.add_argument('--point-size', type=float, default=2.0, help='raw 模式点大小 (voxel 模式会略加 1)')
    p.add_argument('--no-axis-lines', action='store_true', help='隐藏自绘轴线 (默认显示定制轴线)')
    p.add_argument('--endpoint', default=None, help='终点坐标 x,y,z；可带括号，如 0.55271,-0.0752,-0.0828 或 (0.55271,-0.0752,-0.0828)')
    return p.parse_args()

# '(0.5577,-0.0674,-0.0830)'
def main():
    args = parse_args()
    np.random.seed(args.seed)
    # 透明度归一化
    alpha = max(0.05, min(1.0, float(args.alpha)))

    # 解析 endpoint，兼容带括号/方括号/花括号形式
    endpoint_xyz = None
    # if args.endpoint:
    #     raw_ep = args.endpoint.strip()
    #     # 去掉包裹整体的括号或中括号/花括号
    #     if (raw_ep[0] in '([{') and (raw_ep[-1] in ')]}') and len(raw_ep) > 2:
    #         raw_ep = raw_ep[1:-1]
    #     # 去除内部可能的空格
    #     raw_ep = raw_ep.replace(' ', '')
    #     parts = raw_ep.split(',')
    #     if len(parts) == 3:
    #         try:
    #             endpoint_xyz = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
    #         except Exception:
    #             print(f"[WARN] endpoint 解析失败: {args.endpoint}")
    #     else:
    #         print(f"[WARN] endpoint 参数格式应为 x,y,z: {args.endpoint}")

    episodes_filter = None
    if args.episodes:
        episodes_filter = parse_episode_filter(args.episodes)
        print(f"过滤 episode 列表: {episodes_filter}")
    paths: List[str] = args.path
    multi_path_mode = len(paths) > 1
    multi_dataset_mode = (args.datasets is not None) and not multi_path_mode

    if multi_path_mode and args.datasets:
        print('警告：检测到同时提供多个 --path 与 --datasets；将忽略 --datasets，按多 path 模式处理。')

    if multi_path_mode:
        print(f"多 path 模式，总共 {len(paths)} 个来源：")
        for i, pth in enumerate(paths):
            print(f"  [{i}] {pth}")
        points_dict = {}
        stats_dict = {}
        all_list = []
        for i, pth in enumerate(paths):
            print(f"\n== 收集来源[{i}] {pth} ==")
            pts, dsname, stats = collect_points_single_dataset(
                path=pth,
                h5_name=args.h5_name,
                dataset=args.dataset,  # 每个 path 可独立自动检测
                episodes_filter=episodes_filter,
                per_episode_max=args.per_episode_max,
                global_max=args.global_max_points,
            )
            label = os.path.basename(os.path.abspath(pth.rstrip('/')))
            if not label:
                label = f'source{i}'
            # 防止重名
            base_label = label
            k = 1
            while label in points_dict:
                label = f"{base_label}_{k}"; k += 1
            points_dict[label] = pts
            stats_dict[label] = stats
            all_list.append(pts)
            print(f"  -> 收集 {len(pts)} 点 (label={label})")
        all_points_concat = np.concatenate(all_list, axis=0)
        print(f"\n合并全部来源点数: {len(all_points_concat)}")
    elif multi_dataset_mode:
        datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
        if not datasets:
            raise ValueError('--datasets 提供为空。')
        print(f"多数据集模式 (单 path): {datasets}")
        # 此时必须只有一个 path
        points_dict, stats_dict = collect_points_multi_datasets(
            path=paths[0],
            h5_name=args.h5_name,
            datasets=datasets,
            episodes_filter=episodes_filter,
            per_episode_max=args.per_episode_max,
            global_max=args.global_max_points,
        )
        all_points_concat = np.concatenate(list(points_dict.values()), axis=0)
    else:
        print("单 path + 单数据集模式。开始收集点 ...")
        pts, dsname, stats = collect_points_single_dataset(
            path=paths[0],
            h5_name=args.h5_name,
            dataset=args.dataset,
            episodes_filter=episodes_filter,
            per_episode_max=args.per_episode_max,
            global_max=args.global_max_points,
        )
        points_dict = {dsname: pts}
        stats_dict = {dsname: stats}
        all_points_concat = pts
        print(f"数据集 {dsname}: {len(pts)} 点。")

    # RAW 模式：直接展示所有点（按来源分类并采样）
    if args.display_mode == 'raw':
        if multi_path_mode:
            order_list = list(points_dict.keys())
        elif multi_dataset_mode:
            order_list = [d for d in [d.strip() for d in (args.datasets or '').split(',')] if d in points_dict]
        else:
            order_list = list(points_dict.keys())
        concat_pts = []
        concat_src = []
        for i, name in enumerate(order_list):
            pts = points_dict[name]
            if len(pts)==0:
                continue
            concat_pts.append(pts)
            concat_src.append(np.full(len(pts), i, dtype=int))
        if not concat_pts:
            print('RAW 模式：没有可显示的点。')
            return
        all_pts = np.concatenate(concat_pts, axis=0)
        all_src = np.concatenate(concat_src, axis=0)
        if args.raw_max_points>0 and len(all_pts)>args.raw_max_points:
            sel = np.random.choice(len(all_pts), args.raw_max_points, replace=False)
            all_pts = all_pts[sel]
            all_src = all_src[sel]

        def gen_gradient(k: int):
            if k<=0: return []
            start = np.array([0xB5,0xEC,0xB5]); end = np.array([0x00,0x4D,0x00]); gamma=0.85
            cols=[]
            for i in range(k):
                t_lin=0 if k==1 else i/(k-1); t=t_lin**gamma
                rgb=(start*(1-t)+end*t).astype(int)
                cols.append('#'+''.join(f'{v:02x}' for v in rgb))
            return cols
        def parse_colors(s: str, k:int):
            hexdigits = set(string.hexdigits.lower())
            raw = [c.strip() for c in s.split(',')] if s else []
            cleaned = []
            for c in raw:
                if not c:
                    continue
                original = c
                # 去掉多余前缀 '#'
                while c.startswith('##'):
                    c = c[1:]
                if c.startswith('#'):
                    c_body = c[1:]
                else:
                    c_body = c
                # 若纯 6 位 HEX 则标准化
                if len(c_body)==6 and all(ch.lower() in hexdigits for ch in c_body):
                    c = '#' + c_body.lower()
                else:
                    # 非法则跳过
                    continue
                cleaned.append(c)
            if not cleaned:
                cleaned = ['#8fbcd4','#b4d8c2','#f6c9c9','#ded2eb','#f1ddb7']  # 柔和默认
            if len(cleaned)<k:
                cleaned=(cleaned*((k+len(cleaned)-1)//len(cleaned)))[:k]
            return cleaned[:k]

        color_mode = args.color_mode.strip().lower()
        # 仅当单来源时才强制绿色，否则保持用户选择
        if (len(order_list)<=1) and color_mode!='green':
            print('[INFO] 单一来源，颜色模式自动改为 green')
            color_mode='green'
        if color_mode not in ('green','gradient','category'):
            print(f'[WARN] 未知颜色模式 {color_mode}，回退 green')
            color_mode='green'

        ps = max(0.5, args.point_size)
        if color_mode=='green':
            traces=[go.Scatter3d(x=all_pts[:,0],y=all_pts[:,1],z=all_pts[:,2],mode='markers',name='<b>points</b>',marker=dict(size=ps,color='green',opacity=alpha))]
        else:
            if color_mode=='gradient':
                palette = gen_gradient(len(order_list))
            else:  # category
                palette = parse_colors(args.colors, len(order_list))
            print(f'[DEBUG] raw 模式 color_mode={color_mode}, palette={palette}')
            traces=[]
            legend_traces=[]
            for i,name in enumerate(order_list):
                sel = all_pts[all_src==i]
                if not len(sel):
                    continue
                display_name = name
                if color_mode=='category' and len(order_list)==3:
                    display_name = CATEGORY_DIVERSITY_LABELS[i]
                traces.append(go.Scatter3d(x=sel[:,0],y=sel[:,1],z=sel[:,2],mode='markers',
                                           name=f'_raw_hidden_{i}', showlegend=False,
                                           marker=dict(size=ps,color=palette[i],opacity=alpha)))
                legend_size=max(int(ps*LEGEND_MARKER_SCALE), LEGEND_MARKER_MIN)
                legend_traces.append(go.Scatter3d(x=[sel[0,0]],y=[sel[0,1]],z=[sel[0,2]],mode='markers',
                                                  name=f'<b>{display_name}</b>',
                                                  marker=dict(size=legend_size,color=palette[i],opacity=1.0),
                                                  showlegend=True))
            traces.extend(legend_traces)
        fig = go.Figure(data=traces)
        axis_style = dict(showbackground=False, backgroundcolor='rgba(0,0,0,0)', showgrid=False, zeroline=False, showticklabels=False)
        fig.update_layout(
            title='',
            scene=dict(xaxis={**axis_style, 'showline':True, 'ticks':''},
                       yaxis={**axis_style, 'showline':True, 'ticks':''},
                       zaxis={**axis_style, 'showline':True, 'ticks':''},
                       bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(yanchor='top',y=0.99,xanchor='left',x=0.01, font=dict(size=16))
        )

        # endpoint 星形 (使用 cross + x 叠加形成星号效果)
        if endpoint_xyz is not None:
            ex, ey, ez = endpoint_xyz.tolist()
            star_color = '#ff4444'
            base_size = max(ps*4.0, 14)
            # 大号实际显示（不进图例）
            fig.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode='markers',
                                       marker=dict(size=base_size, color=star_color, symbol='cross', opacity=1.0),
                                       showlegend=False))
            fig.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode='markers',
                                       marker=dict(size=base_size*0.7, color=star_color, symbol='x', opacity=1.0),
                                       showlegend=False))
            # 小号用于图例
            fig.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode='markers',
                                       marker=dict(size=6, color=star_color, symbol='x', opacity=1.0),
                                       name='endpoint'))

        if not args.no_axis_lines:
            # 自绘 3 根轴线 (通过数据包围盒中心)
            min_xyz = all_pts.min(axis=0)
            max_xyz = all_pts.max(axis=0)
            center = (min_xyz + max_xyz) / 2.0
            cx, cy, cz = center
            # X 轴
            fig.add_trace(go.Scatter3d(x=[min_xyz[0], max_xyz[0]], y=[cy, cy], z=[cz, cz], mode='lines',
                                       line=dict(color='#666', width=4), name='X'))
            fig.add_trace(go.Scatter3d(x=[max_xyz[0]], y=[cy], z=[cz], mode='text', text=['X'],
                                       textposition='top center', showlegend=False,
                                       textfont=dict(size=AXIS_LETTER_FONT_SIZE)))
            # Y 轴
            fig.add_trace(go.Scatter3d(x=[cx, cx], y=[min_xyz[1], max_xyz[1]], z=[cz, cz], mode='lines',
                                       line=dict(color='#666', width=4), name='Y'))
            fig.add_trace(go.Scatter3d(x=[cx], y=[max_xyz[1]], z=[cz], mode='text', text=['Y'],
                                       textposition='top center', showlegend=False,
                                       textfont=dict(size=AXIS_LETTER_FONT_SIZE)))
            # Z 轴
            fig.add_trace(go.Scatter3d(x=[cx, cx], y=[cy, cy], z=[min_xyz[2], max_xyz[2]], mode='lines',
                                       line=dict(color='#666', width=4), name='Z'))
            fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[max_xyz[2]], mode='text', text=['Z'],
                                       textposition='top center', showlegend=False,
                                       textfont=dict(size=AXIS_LETTER_FONT_SIZE)))
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fig.write_html(args.output)

    # 生成 2D 投影图 (平面正交)
        plane = args.raw_2d_plane
        if args.raw_2d_output:
            img_path = args.raw_2d_output
        else:
            root, ext = os.path.splitext(args.output)
            img_path = root + '_2d.png'
        if plane == 'xy':
            ax0, ax1, xlabel, ylabel = 0, 1, 'X', 'Y'
        elif plane == 'xz':
            ax0, ax1, xlabel, ylabel = 0, 2, 'X', 'Z'
        else:  # yz
            ax0, ax1, xlabel, ylabel = 1, 2, 'Y', 'Z'
        plt.figure(figsize=(6,6), dpi=160)
        if color_mode=='green':
            plt.scatter(all_pts[:,ax0], all_pts[:,ax1], c='green', s=2, alpha=alpha, linewidths=0, label='points')
        else:
            for i,name in enumerate(order_list):
                sel = all_pts[all_src==i]
                if not len(sel):
                    continue
                plt.scatter(sel[:,ax0], sel[:,ax1], c=palette[i], s=2, alpha=alpha, linewidths=0, label=name)
        if endpoint_xyz is not None:
            ex, ey, ez = endpoint_xyz.tolist()
            if plane == 'xy': px, py = ex, ey
            elif plane == 'xz': px, py = ex, ez
            else: px, py = ey, ez
            # 大号星标（无 label）
            plt.scatter([px], [py], c='#ff4444', s=180, marker='*', edgecolors='none')
            # 小号星标进入图例
            plt.scatter([px], [py], c='#ff4444', s=40, marker='*', edgecolors='none', label='endpoint')
        if (color_mode!='green' and len(order_list)>1) or endpoint_xyz is not None:
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            from matplotlib.lines import Line2D
            new_handles=[]; new_labels=[]
            for h,lab in zip(handles, labels):
                col=None
                if hasattr(h,'get_facecolor'):
                    fc=h.get_facecolor(); col=fc[0] if len(fc) else None
                if col is None and hasattr(h,'get_color'):
                    col=h.get_color()
                if color_mode=='category' and len(order_list)==3 and lab in order_list:
                    idx=order_list.index(lab); lab_disp=CATEGORY_DIVERSITY_LABELS[idx]
                else:
                    lab_disp=lab
                if col is None: col='gray'
                marker_style = '*' if lab_disp.lower()=='endpoint' else 'o'
                new_handles.append(Line2D([0],[0], marker=marker_style, linestyle='None', markersize=10,
                                          markerfacecolor=col, markeredgecolor=col))
                new_labels.append(lab_disp)
            ax.legend(new_handles, new_labels, frameon=False, loc='best', prop={'size': LEGEND_FONT_SIZE, 'weight':'bold'})
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE, fontweight='bold')
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path)
        # 同步输出 PDF
        pdf_2d = os.path.splitext(img_path)[0] + '.pdf'
        plt.savefig(pdf_2d)
        plt.close()

        # 生成 3D 视角投影 (通过旋转点再 2D 渲染，模拟固定相机)
        if args.raw_3d_output:
            img3d_path = args.raw_3d_output
        else:
            root, ext = os.path.splitext(args.output)
            img3d_path = root + '_3d.png'
        elev = np.deg2rad(args.raw_3d_elev)
        azim = np.deg2rad(args.raw_3d_azim)
        # 构造旋转矩阵 (先绕 Z azim，再绕 X elev)
        cosA, sinA = np.cos(azim), np.sin(azim)
        Rz = np.array([[cosA, -sinA, 0],[sinA, cosA, 0],[0,0,1]])
        cosE, sinE = np.cos(elev), np.sin(elev)
        Rx = np.array([[1,0,0],[0,cosE,-sinE],[0,sinE,cosE]])
        R = Rz @ Rx
        prot = all_pts @ R  # shape (N,3)
        x2d = prot[:,0]; y2d = prot[:,1]; depth = prot[:,2]
        # 深度排序 (由远到近绘制)
        order = np.argsort(depth)
        x2d, y2d = x2d[order], y2d[order]
        if color_mode=='green':
            cols = np.array(['green']*len(all_pts))[order]
        else:
            cols = np.array([palette[i] for i in all_src])[order]
        # 归一化点尺寸可选: 随深度稍微缩小 (可调)
        # size_factor = 1.0 - 0.15*(depth[order]-depth.min())/(depth.max()-depth.min()+1e-9)
        plt.figure(figsize=tuple(args.raw_3d_size), dpi=170)
        plt.scatter(x2d, y2d, c=cols, s=2, alpha=alpha, linewidths=0)
        if endpoint_xyz is not None:
            exr, eyr, ezr = (endpoint_xyz @ R).tolist()
            plt.scatter([exr], [eyr], c='#ff4444', s=140, marker='*', edgecolors='none')
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout(pad=0)
        os.makedirs(os.path.dirname(img3d_path), exist_ok=True)
        plt.savefig(img3d_path, bbox_inches='tight', pad_inches=0)
        pdf_3d = os.path.splitext(img3d_path)[0] + '.pdf'
        plt.savefig(pdf_3d, bbox_inches='tight', pad_inches=0)
        plt.close()
        print('\n============= RAW 模式统计 =============')
        print(f'显示点数: {len(all_pts)} (限制 {args.raw_max_points})')
        for i,name in enumerate(order_list):
            print(f'  {i}. {name}: {(all_src==i).sum()} 点')
        print(f'输出文件: {os.path.abspath(args.output)}')
        print(f'2D 投影: {os.path.abspath(img_path)}  (plane={plane})')
        print(f'3D 视角投影: {os.path.abspath(img3d_path)}  (elev={args.raw_3d_elev}, azim={args.raw_3d_azim})')
        # 生成每个来源的单独拆分图
        if args.raw_individual_dir and len(order_list) > 1:
            out_dir = args.raw_individual_dir
            os.makedirs(out_dir, exist_ok=True)
            # 统一轴范围: 使用 all_pts 的 min/max
            global_min = all_pts.min(axis=0)
            global_max = all_pts.max(axis=0)
            # 为 2D 投影和 3D 投影重复逻辑封装函数
            elev = np.deg2rad(args.raw_3d_elev)
            azim = np.deg2rad(args.raw_3d_azim)
            cosA, sinA = np.cos(azim), np.sin(azim)
            Rz = np.array([[cosA, -sinA, 0],[sinA, cosA, 0],[0,0,1]])
            cosE, sinE = np.cos(elev), np.sin(elev)
            Rx = np.array([[1,0,0],[0,cosE,-sinE],[0,sinE,cosE]])
            R = Rz @ Rx
            for i, name in enumerate(order_list):
                src_mask = (all_src == i)
                pts_src = all_pts[src_mask]
                if pts_src.size == 0:
                    continue
                # 主 3D 拆分 HTML
                if color_mode == 'green':
                    src_color = 'green'
                else:
                    src_color = palette[i]
                ps_src = ps
                disp_name = name
                if color_mode=='category' and len(order_list)==3:
                    disp_name = CATEGORY_DIVERSITY_LABELS[i]
                fig_src = go.Figure(data=[go.Scatter3d(x=pts_src[:,0], y=pts_src[:,1], z=pts_src[:,2],
                                                       mode='markers', name=f'<b>{disp_name}</b>',
                                                       marker=dict(size=ps_src, color=src_color, opacity=alpha))])
                axis_style2 = dict(showbackground=False, backgroundcolor='rgba(0,0,0,0)', showgrid=False, zeroline=False, showticklabels=False)
                fig_src.update_layout(
                    title='',
                    scene=dict(xaxis={**axis_style2,'showline':True,'ticks':'','range':[global_min[0],global_max[0]]},
                               yaxis={**axis_style2,'showline':True,'ticks':'','range':[global_min[1],global_max[1]]},
                               zaxis={**axis_style2,'showline':True,'ticks':'','range':[global_min[2],global_max[2]]},
                               bgcolor='rgba(0,0,0,0)'),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                # endpoint 保持一致
                if endpoint_xyz is not None:
                    ex, ey, ez = endpoint_xyz.tolist()
                    base_size_src = max(ps_src*4.0, 14)
                    fig_src.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode='markers',
                                                   marker=dict(size=base_size_src, color='#ff4444', symbol='cross', opacity=1.0),
                                                   showlegend=False))
                    fig_src.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode='markers',
                                                   marker=dict(size=base_size_src*0.7, color='#ff4444', symbol='x', opacity=1.0),
                                                   showlegend=False))
                    fig_src.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode='markers',
                                                   marker=dict(size=6, color='#ff4444', symbol='x', opacity=1.0),
                                                   name='endpoint'))
                # 可选轴线
                if not args.no_axis_lines:
                    cx, cy, cz = (global_min + global_max)/2.0
                    fig_src.add_trace(go.Scatter3d(x=[global_min[0], global_max[0]], y=[cy, cy], z=[cz, cz], mode='lines', line=dict(color='#666', width=4), name='X'))
                    fig_src.add_trace(go.Scatter3d(x=[global_max[0]], y=[cy], z=[cz], mode='text', text=['X'], showlegend=False, textfont=dict(size=AXIS_LETTER_FONT_SIZE)))
                    fig_src.add_trace(go.Scatter3d(x=[cx, cx], y=[global_min[1], global_max[1]], z=[cz, cz], mode='lines', line=dict(color='#666', width=4), name='Y'))
                    fig_src.add_trace(go.Scatter3d(x=[cx], y=[global_max[1]], z=[cz], mode='text', text=['Y'], showlegend=False, textfont=dict(size=AXIS_LETTER_FONT_SIZE)))
                    fig_src.add_trace(go.Scatter3d(x=[cx, cx], y=[cy, cy], z=[global_min[2], global_max[2]], mode='lines', line=dict(color='#666', width=4), name='Z'))
                    fig_src.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[global_max[2]], mode='text', text=['Z'], showlegend=False, textfont=dict(size=AXIS_LETTER_FONT_SIZE)))
                indiv_html = os.path.join(out_dir, f'{i:02d}_{name}_raw.html')
                fig_src.write_html(indiv_html)
                # 2D projection for this source
                if plane == 'xy':
                    ax0, ax1, xlabel, ylabel = 0,1,'X','Y'
                elif plane == 'xz':
                    ax0, ax1, xlabel, ylabel = 0,2,'X','Z'
                else:
                    ax0, ax1, xlabel, ylabel = 1,2,'Y','Z'
                plt.figure(figsize=(6,6), dpi=160)
                plt.scatter(pts_src[:,ax0], pts_src[:,ax1], c=src_color, s=2, alpha=alpha, linewidths=0)
                plt.xlim(global_min[ax0], global_max[ax0])
                plt.ylim(global_min[ax1], global_max[ax1])
                if endpoint_xyz is not None:
                    ex, ey, ez = endpoint_xyz.tolist()
                    if plane == 'xy': px, py = ex, ey
                    elif plane == 'xz': px, py = ex, ez
                    else: px, py = ey, ez
                    plt.scatter([px], [py], c='#ff4444', s=120, marker='*', edgecolors='none')
                    plt.scatter([px], [py], c='#ff4444', s=40, marker='*', edgecolors='none', label='endpoint')
                if endpoint_xyz is not None or (color_mode!='green' and len(order_list)>1):
                    ax_s = plt.gca()
                    h_s, l_s = ax_s.get_legend_handles_labels()
                    if color_mode=='category' and len(order_list)==3:
                        mapped_s = []
                        for lab in l_s:
                            if lab in order_list:
                                mapped_s.append(CATEGORY_DIVERSITY_LABELS[order_list.index(lab)])
                            else:
                                mapped_s.append(lab)
                        l_s = mapped_s
                    # 替换成自定义代理，endpoint 保持星形
                    from matplotlib.lines import Line2D
                    new_h=[]; new_l=[]
                    for hh,lab in zip(h_s,l_s):
                        col=None
                        if hasattr(hh,'get_facecolor'):
                            fc=hh.get_facecolor(); col=fc[0] if len(fc) else None
                        if col is None and hasattr(hh,'get_color'):
                            col=hh.get_color()
                        disp=lab
                        if color_mode=='category' and len(order_list)==3 and lab in order_list:
                            disp=CATEGORY_DIVERSITY_LABELS[order_list.index(lab)]
                        marker_style='*' if disp.lower()=='endpoint' else 'o'
                        if col is None: col='gray'
                        new_h.append(Line2D([0],[0], marker=marker_style, linestyle='None', markersize=9,
                                             markerfacecolor=col, markeredgecolor=col))
                        new_l.append(disp)
                    ax_s.legend(new_h, new_l, frameon=False, loc='best', prop={'size': LEGEND_FONT_SIZE})
                plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE, fontweight='bold'); plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight='bold')
                plt.tight_layout()
                indiv_2d = os.path.join(out_dir, f'{i:02d}_{name}_2d.png')
                plt.savefig(indiv_2d)
                indiv_2d_pdf = os.path.splitext(indiv_2d)[0] + '.pdf'
                plt.savefig(indiv_2d_pdf)
                plt.close()
                # 3D viewpoint projection
                prot_src = pts_src @ R
                x2d_s = prot_src[:,0]; y2d_s = prot_src[:,1]; depth_s = prot_src[:,2]
                order_s = np.argsort(depth_s)
                x2d_s = x2d_s[order_s]; y2d_s = y2d_s[order_s]
                plt.figure(figsize=tuple(args.raw_3d_size), dpi=170)
                plt.scatter(x2d_s, y2d_s, c=src_color, s=2, alpha=alpha, linewidths=0)
                if endpoint_xyz is not None:
                    exr, eyr, ezr = (endpoint_xyz @ R).tolist()
                    plt.scatter([exr], [eyr], c='#ff4444', s=140, marker='*', edgecolors='none')
                plt.axis('equal'); plt.axis('off'); plt.tight_layout(pad=0)
                indiv_3d = os.path.join(out_dir, f'{i:02d}_{name}_3d.png')
                plt.savefig(indiv_3d, bbox_inches='tight', pad_inches=0)
                indiv_3d_pdf = os.path.splitext(indiv_3d)[0] + '.pdf'
                plt.savefig(indiv_3d_pdf, bbox_inches='tight', pad_inches=0)
                plt.close()
            print(f'单独拆分输出目录: {os.path.abspath(out_dir)}')
        print('========================================\n')
        return

    print("计算包围盒与体素网格……")
    # 体素模式：继续

    grid_info = compute_voxel_grid(all_points_concat, beta=args.beta, max_total_voxels=args.max_total_voxels)
    nx, ny, nz = grid_info['counts']
    print(f"体素网格大小: {nx} x {ny} x {nz} = {grid_info['total_voxels']} (voxel_size={grid_info['voxel_size']:.6f})")

    # 多/单数据集占用
    def _compute_occ(pts):
        return occupancy_by_binning(pts, grid_info) if args.method == 'bin' else occupancy_by_nn(pts, grid_info)

    occ_map: Dict[str, np.ndarray] = {}
    for label_name, pts in points_dict.items():
        if len(pts) == 0:
            continue
        occ_map[label_name] = _compute_occ(pts)
        print(f"来源 {label_name}: 占用体素 {occ_map[label_name].sum()} / {grid_info['total_voxels']}")

    nx, ny, nz = grid_info['counts']
    label = np.full((nx, ny, nz), -1, dtype=int)

    if multi_path_mode:
        order_list = list(points_dict.keys())  # 按路径顺序
    elif multi_dataset_mode:
        # 使用用户给定 datasets 顺序 (已在上方保留) 但需要转换为实际 key (数据集名称)
        order_list = [d for d in [d.strip() for d in args.datasets.split(',')] if d in occ_map]
    else:
        order_list = list(points_dict.keys())

    # 覆盖策略：按提供顺序逐个染色，后出现的来源直接覆盖之前的标签
    for i, name in enumerate(order_list):
        mask = occ_map[name]
        label[mask] = i

    # 颜色与可视化
    xs, ys, zs = grid_info['xs'], grid_info['ys'], grid_info['zs']
    Xc, Yc, Zc = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1)
    labels_flat = label.ravel()
    occupied_mask = labels_flat >= 0
    occupied_points = centers[occupied_mask]

    traces = []

    def gen_gradient(k: int):
        if k <= 0: return []
        # 调整起始色为略深的浅绿 (原 #D6F5D6 -> 现 #B5ECB5) 防止过浅难以看见
        # 并使用 gamma 校正让中间色更均衡
        start = np.array([0xB5, 0xEC, 0xB5])  # 新浅绿稍深
        end = np.array([0x00, 0x4D, 0x00])    # 深绿
        gamma = 0.85  # <1 拉伸浅色段，使早期颜色更接近深一点
        cols = []
        for i in range(k):
            t_lin = 0 if k==1 else i/(k-1)
            t = t_lin ** gamma
            rgb = (start*(1-t)+end*t).astype(int)
            cols.append('#' + ''.join(f'{v:02x}' for v in rgb))
        return cols

    def parse_colors(s: str, k: int):
        hexdigits = set(string.hexdigits.lower())
        raw = [c.strip() for c in s.split(',')] if s else []
        cleaned=[]
        for c in raw:
            if not c:
                continue
            while c.startswith('##'):
                c=c[1:]
            if c.startswith('#'):
                body=c[1:]
            else:
                body=c
            if len(body)==6 and all(ch.lower() in hexdigits for ch in body):
                cleaned.append('#'+body.lower())
        if not cleaned:
            cleaned=['#8fbcd4','#b4d8c2','#f6c9c9','#ded2eb','#f1ddb7']
        if len(cleaned)<k:
            cleaned=(cleaned*((k+len(cleaned)-1)//len(cleaned)))[:k]
        return cleaned[:k]

    color_mode = args.color_mode
    if (len(order_list) <= 1) and color_mode != 'green':
        # 单来源时强制统一
        color_mode = 'green'

    if color_mode == 'green':
        occ_ps = max(0.5, args.point_size + 1.0)
        traces.append(go.Scatter3d(
            x=occupied_points[:, 0], y=occupied_points[:, 1], z=occupied_points[:, 2],
            mode='markers', name='占用体素',
            marker=dict(size=occ_ps, color='green', opacity=alpha)
        ))
    else:
        if color_mode == 'gradient':
            palette = gen_gradient(len(order_list))
        else:  # category
            palette = parse_colors(args.colors, len(order_list))
        for i, name in enumerate(order_list):
            sel = centers[labels_flat == i]
            if not len(sel):
                continue
            occ_ps = max(0.5, args.point_size + 1.0)
            disp_name = name
            if color_mode=='category' and len(order_list)==3:
                disp_name = CATEGORY_DIVERSITY_LABELS[i]
            traces.append(go.Scatter3d(
                x=sel[:,0], y=sel[:,1], z=sel[:,2],
                mode='markers', name=f'<b>{disp_name}</b>',
                marker=dict(size=occ_ps, color=palette[i], opacity=alpha)
            ))

    total_voxels = grid_info['total_voxels']
    occupied_total = (labels_flat >= 0).sum()
    occ_ratio = occupied_total / total_voxels if total_voxels else 0

    title_lines = [
        f"多数据集体素覆盖: {occupied_total}/{total_voxels} ( {occ_ratio*100:.2f}% )",
        f"voxel_size={grid_info['voxel_size']:.4g}, grid={grid_info['counts']}"
    ]
    fig = go.Figure(data=traces)
    axis_style = dict(showbackground=False, backgroundcolor='rgba(0,0,0,0)', showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        title='',  # 去除标题
        scene=dict(xaxis={**axis_style,'showline':True,'ticks':''}, yaxis={**axis_style,'showline':True,'ticks':''}, zaxis={**axis_style,'showline':True,'ticks':''}, bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )
    # voxel 模式 endpoint
    if endpoint_xyz is not None:
        ex, ey, ez = endpoint_xyz.tolist()
        star_color = '#ff4444'
        occ_ps = max(0.5, args.point_size + 1.0)
        base_size = max(occ_ps*4.0, 16)
        # 大号显示（不进图例）
        fig.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode='markers',
                                   marker=dict(size=base_size, color=star_color, symbol='cross', opacity=1.0),
                                   showlegend=False))
        fig.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode='markers',
                                   marker=dict(size=base_size*0.7, color=star_color, symbol='x', opacity=1.0),
                                   showlegend=False))
    # 小号 legend
    fig.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[ez], mode='markers',
                   marker=dict(size=6, color=star_color, symbol='x', opacity=1.0),
                   name='endpoint'))
    if not args.no_axis_lines:
        min_xyz = occupied_points.min(axis=0) if len(occupied_points) else np.array([0,0,0])
        max_xyz = occupied_points.max(axis=0) if len(occupied_points) else np.array([1,1,1])
        center = (min_xyz + max_xyz)/2.0
        cx, cy, cz = center
        fig.add_trace(go.Scatter3d(x=[min_xyz[0], max_xyz[0]], y=[cy, cy], z=[cz, cz], mode='lines', line=dict(color='#666', width=4), name='X轴'))
        fig.add_trace(go.Scatter3d(x=[max_xyz[0]], y=[cy], z=[cz], mode='text', text=['X'], showlegend=False, textfont=dict(size=AXIS_LETTER_FONT_SIZE)))
        fig.add_trace(go.Scatter3d(x=[cx, cx], y=[min_xyz[1], max_xyz[1]], z=[cz, cz], mode='lines', line=dict(color='#666', width=4), name='Y轴'))
        fig.add_trace(go.Scatter3d(x=[cx], y=[max_xyz[1]], z=[cz], mode='text', text=['Y'], showlegend=False, textfont=dict(size=AXIS_LETTER_FONT_SIZE)))
        fig.add_trace(go.Scatter3d(x=[cx, cx], y=[cy, cy], z=[min_xyz[2], max_xyz[2]], mode='lines', line=dict(color='#666', width=4), name='Z轴'))
        fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[max_xyz[2]], mode='text', text=['Z'], showlegend=False, textfont=dict(size=AXIS_LETTER_FONT_SIZE)))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.write_html(args.output)

    print("\n================ 统计信息 ================")
    print(f"总体素数: {total_voxels}")
    print(f"被占用体素数: {occupied_total} (占用率 {occ_ratio*100:.2f}%)")
    # 统计各来源（顺序保留）
    for i, ds in enumerate(order_list):
        ds_occ = (labels_flat == i).sum()
        print(f"  {i}. {ds}: {ds_occ} 体素; 占全部 {ds_occ/total_voxels*100:.2f}%")
    print(f"输出文件: {os.path.abspath(args.output)}")
    print("==========================================\n")


if __name__ == '__main__':
    main()

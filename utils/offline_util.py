"""offline_util
=================================
基于已录制 episode 的状态轨迹 (state/eef/position) 与 endpoint 计算离线奖励。

设计背景
---------
在 `generate.py` 中，轨迹的第 6~9 列(索引 6:9) 对应执行末端沿某条曲线 (bezier / cone) 的 XYZ 序列。
你现在的 HDF5 (aligned_joints.h5) 结构中：

aligned_joints.h5
├── state/eef/position      shape (T, 12)
├── action/eef/position     shape (T, 12)
└── endpoint                shape (3,)  (目标 XYZ)

要求的奖励: reward_t = 1 - D_remain(state_t, endpoint) / D_total
其中 D_remain 为沿轨迹(曲线)从当前状态位置到终点的“曲线剩余长度”，D_total 为整条曲线总长度。

如果没有显式保存原曲线控制点，我们可以直接把已录制的 (curve segment) 序列当作曲线离散采样：
  points_t = state/eef/position[:, 6:9]
这样：
  D_total = sum(||p_{i+1}-p_i|| for i=0..T-2)
  D_remain(t) = sum(||p_{i+1}-p_i|| for i=t..T-2)
  reward_t = 1 - D_remain(t) / max(D_total, eps)
  并强制 reward_{T-1} = 1.0 (数值稳定 & 直观结束奖励)

若 D_total≈0 (静止轨迹)，则返回全 0，最后一步 1。

提供的函数
-----------
1. compute_progress_rewards(points)
   输入 (T,3) 位置序列，输出 (T,) 奖励数组。

2. compute_rewards_from_state(state_pos, endpoint, pos_slice=(6,9))
   从 state/eef/position 数组抽取位置列 (默认 6:9) 并验证终点，然后计算奖励。

3. offline_reward(state_eef, action_eef, endpoint, pos_slice=(6,9))
   对外主接口；action 当前未使用（预留可扩展），返回奖励 (T,) float32。

4. compute_rewards_from_h5(h5_path, pos_slice=(6,9))
   直接读取单个 aligned_joints.h5 计算奖励。

使用示例
--------
>>> import h5py
>>> from utils.offline_util import compute_rewards_from_h5
>>> rewards = compute_rewards_from_h5('episode0/aligned_joints.h5')
>>> rewards.shape
(T,)

注意
----
如果将来 endpoint 也包含 RPY 或更多维度，可在本工具中扩展：
 - 可通过额外参数传入 rpy_end，再用 generate.get_direct() & 曲线重建以插值更细轨迹。
 - 当前实现基于离散采样路径，足够用于比例进度奖励。
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import h5py
import os

__all__ = [
    'compute_progress_rewards',
    'compute_rewards_from_state',
    'offline_reward',
    'compute_rewards_from_h5'
]

def _path_segment_lengths(points: np.ndarray) -> np.ndarray:
    """返回相邻点欧氏距离数组 shape (T-1,)."""
    return np.linalg.norm(points[1:] - points[:-1], axis=1)


def compute_progress_rewards(points: np.ndarray,
                              endpoint: np.ndarray | None = None,
                              ensure_terminal_one: bool = True,
                              curve_type: str = 'bezier',  # 保留签名兼容
                              beta: float = 0.003,
                              chunk_size: int = 35,
                              use_generated_curve: bool = True,
                              rpy_seq: np.ndarray | None = None,
                              rpy_end: np.ndarray | None = None,
                              ori_weight: float = 0.0) -> np.ndarray:
    """根据最新规范计算奖励：

    对每一个点 p_i 单独：
        1. 生成一条从 p_i 到 endpoint_pos 的曲线:
             curve_i = generate_curve(curve_type, p_i, endpoint_pos, get_direct(endpoint_pos, endpoint_rpy), beta)
        2. 距离奖励距离部分:
             L_i = \sum ||curve_i[k+1]-curve_i[k]||
             dis_base_i = 1 - L_i
             方向余弦 cos_dir_i = cos( (p_{i+1}-p_i), (curve_i[1]-curve_i[0]) )  (i==T-1 时取 1)
             dis_reward_i = dis_base_i * cos_dir_i
        3. 姿态奖励 ori_reward_i：
             - 生成局部姿态参考 rpy_curve_i = generate_rpy_trajectory(rpy_seq[i], endpoint_rpy, 2)
                 取 v_ori_ref = rpy_curve_i[1]-rpy_curve_i[0]
                 v_ori_real = rpy_seq[i+1]-rpy_seq[i] (末帧 cos=1)
                 cos_ori_dir_i = cos(v_ori_real, v_ori_ref)
             - 角差 θ_i (四元数) = angle(rpy_seq[i], endpoint_rpy)
                 ori_base_i = 1 - θ_i
             - ori_reward_i = ori_base_i * cos_ori_dir_i
        4. 融合: reward_i = (1 - ori_weight) * dis_reward_i + ori_weight * ori_reward_i

    退化与边界:
        - 若曲线生成失败 -> 使用直线 (p_i, endpoint) 且 cos_dir_i=1。
        - 若 rpy 不可用或失败 -> ori_reward_i=0。
        - 不做归一化，不裁剪负值。
        - ensure_terminal_one: 末帧距离足够小则提升末帧为序列最大值。
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'points 期望形状 (T,3), 得到 {points.shape}')
    T = points.shape[0]
    if T == 0:
        return np.zeros((0,), dtype=np.float32)

    # 解析 endpoint
    if endpoint is None:
        endpoint = points[-1]
    endpoint = np.asarray(endpoint, dtype=np.float32)
    if endpoint.shape[0] < 3:
        raise ValueError('endpoint 至少包含 3 个位置维度')
    endpoint_pos = endpoint[:3]
    endpoint_rpy = endpoint[3:6] if endpoint.shape[0] >= 6 else None

    # 预取姿态序列（如果有）
    rpy_seq = None if rpy_seq is None else np.asarray(rpy_seq, dtype=np.float32)
    endpoint_rpy = endpoint_rpy if rpy_end is None else rpy_end  # 兼容旧参数 rpy_end 但优先 endpoint 中的姿态
    if endpoint_rpy is None and rpy_end is not None:
        endpoint_rpy = rpy_end

    # 输出数组
    dis_rewards = np.zeros((T,), dtype=np.float32)
    ori_rewards = np.zeros((T,), dtype=np.float32)

    # 导入生成函数
    from generate import generate_curve, get_direct
    try:
        from utils.rpy_util import generate_rpy_trajectory  # 用于局部姿态参考
    except Exception:
        generate_rpy_trajectory = None
    from math import acos
    try:
        from scipy.spatial.transform import Rotation as R
        have_rotation = True
    except Exception:
        have_rotation = False

    # 方向向量（用于所有 per-point 曲线的终点方向）
    try:
        if endpoint_rpy is not None:
            direct_vec = get_direct(endpoint_pos, endpoint_rpy)
        else:
            # 若无终点姿态，使用一个默认方向 (0,0,-1)
            direct_vec = np.array([0., 0., -1.], dtype=np.float32)
    except Exception:
        direct_vec = np.array([0., 0., -1.], dtype=np.float32)

    def curve_total_length(curve: np.ndarray) -> float:
        if curve.shape[0] < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(curve[1:] - curve[:-1], axis=1)))

    for i in range(T):
        p_i = points[i]
        # 1) 生成以 p_i 为起点的曲线
        try:
            curve_i = generate_curve(curve_type, p_i, endpoint_pos, direct_vec, beta)
            curve_i = np.asarray(curve_i, dtype=np.float32)
            if curve_i.ndim != 2 or curve_i.shape[1] != 3:
                raise ValueError('curve_i 形状异常')
        except Exception:
            curve_i = np.vstack([p_i, endpoint_pos])  # fallback 直线
        curve_length = len(curve_i)
        if curve_length == 0:
            curve_i = np.vstack([curve_i, endpoint_pos[np.newaxis, :]])
            curve_length = 1

        # gripper = np.tile([0.0, 0.0], (curve_length, 1))
        if curve_length < chunk_size:
            # gripper[-1] = [0.0, 90.0]
            while len(curve_i) < chunk_size:
                curve_i = np.vstack([curve_i, curve_i[-1]])
                # gripper = np.vstack([gripper, gripper[-1]])
            curve_length = len(curve_i)
        elif curve_length > chunk_size:
            curve_i = curve_i[:chunk_size]
            # gripper = gripper[:chunk_size]
            curve_length = chunk_size

        # 距离奖励基值 (1 - 曲线总长度)
        L_i = curve_total_length(curve_i)
        dis_base = 1.0 - L_i

        # 方向余弦 cos(theta_i)
        if i < T - 1 and curve_i.shape[0] >= 2:
            v_real = points[i+1] - p_i
            v_ref = curve_i[1] - curve_i[0]
            nr = np.linalg.norm(v_real)
            nf = np.linalg.norm(v_ref)
            if nf < 1e-8:
                v_ref = np.array(direct_vec, dtype=np.float32)
                nf = np.linalg.norm(v_ref)
            if nr < 1e-8 or nf < 1e-8:
                cos_theta = 1.0
            else:
                cos_theta = float(np.clip(np.dot(v_real, v_ref)/(nr*nf), -1.0, 1.0))
        else:
            cos_theta = 1.0
        dis_rewards[i] = dis_base * cos_theta

        # 姿态奖励 ori_reward_i = (1 - θ_i) * cos_ori_dir_i
        if rpy_seq is not None and endpoint_rpy is not None and have_rotation and i < rpy_seq.shape[0]:
            try:
                r_i = rpy_seq[i]
                q_i = R.from_euler('xyz', r_i).as_quat()
                q_end = R.from_euler('xyz', endpoint_rpy).as_quat()
                dot = float(np.clip(np.abs(np.dot(q_i, q_end)), -1.0, 1.0))
                theta = 2.0 * np.arccos(dot)  # [0,pi]
                ori_base = 1.0 - float(theta)
                # 方向 cos (姿态变化方向 vs 参考姿态方向)
                if i < T-1 and generate_rpy_trajectory is not None:
                    try:
                        rpy_curve = generate_rpy_trajectory(r_i, endpoint_rpy, curve_length)
                        rpy_curve = np.asarray(rpy_curve, dtype=np.float32)
                        v_ori_ref = rpy_curve[1] - rpy_curve[0]
                        v_ori_real = rpy_seq[i+1] - rpy_seq[i]
                        nr = np.linalg.norm(v_ori_real); nf = np.linalg.norm(v_ori_ref)
                        if nr < 1e-8 or nf < 1e-8:
                            cos_ori_dir = 1.0
                        else:
                            cos_ori_dir = float(np.clip(np.dot(v_ori_real, v_ori_ref)/(nr*nf), -1.0, 1.0))
                    except Exception:
                        cos_ori_dir = 1.0
                else:
                    cos_ori_dir = 1.0
                ori_rewards[i] = ori_base * cos_ori_dir
            except Exception:
                ori_rewards[i] = 0.0

    # 融合
    w = float(np.clip(ori_weight, 0.0, 1.0))
    rewards = (1.0 - w) * dis_rewards + w * ori_rewards

    # 末帧提升（若与终点非常接近）
    try:
        if ensure_terminal_one:
            end_dist = np.linalg.norm(points[-1] - endpoint_pos)
            if end_dist < 1e-5:
                rewards[-1] = max(rewards)
    except Exception:
        pass

    return rewards.astype(np.float32)




def compute_rewards_from_state(state_eef_position: np.ndarray, endpoint: np.ndarray | None = None,
                               pos_slice: Tuple[int, int] = (6, 9),
                               rpy_slice: Tuple[int,int] = (9,12),
                               ori_weight: float = 0.0) -> np.ndarray:
    """从完整 state/eef/position (T, D) 中截取 pos_slice 位置列并计算曲线进度奖励。

    参数:
        state_eef_position: (T, D) float 数组 (D>=pos_slice[1])
        endpoint: (3,) 终点 xyz，可为 None (则使用最后一个点)
        pos_slice: 位置列切片 (默认 (6,9)) 对应 generate.py 中的曲线 xyz 部分
    返回:
        (T,) 奖励数组
    """
    state_eef_position = np.asarray(state_eef_position, dtype=np.float32)
    if state_eef_position.ndim != 2:
        raise ValueError('state_eef_position 必须是二维 (T,D)')
    a,b = pos_slice
    if b > state_eef_position.shape[1]:
        raise ValueError(f'pos_slice {pos_slice} 超出维度 {state_eef_position.shape[1]}')
    pts = state_eef_position[:, a:b]
    if pts.shape[1] != 3:
        raise ValueError('位置切片后不是 (T,3)')
    rpy_seq = None
    rpy_end = None
    if rpy_slice[1] <= state_eef_position.shape[1]:
        rpy_seq = state_eef_position[:, rpy_slice[0]:rpy_slice[1]]
    if endpoint is not None:
        endpoint = np.asarray(endpoint, dtype=np.float32)
        if endpoint.shape != (3,):
            raise ValueError('endpoint 期望形状 (3,)')
        # endpoint 不含 rpy_end 时 rpy_end 仍为 None；若你将来扩展 endpoint 包含姿态可传入
    # 当前 endpoint 只含位置 (3,)；若将来 endpoint 扩展为 (6,) 可在此解析 rpy_end = endpoint[3:6]
    return compute_progress_rewards(pts, endpoint=endpoint, rpy_seq=rpy_seq, rpy_end=rpy_end, ori_weight=ori_weight)[:state_eef_position.shape[0]]


def offline_reward(state_eef_position: np.ndarray, action_eef_position: np.ndarray | None,
                   endpoint: np.ndarray | None = None, pos_slice: Tuple[int,int] = (6,9),
                   rpy_slice: Tuple[int,int] = (9,12), ori_weight: float = 0.0) -> np.ndarray:
    """对外主接口：给定 state/action（action 当前不使用，预留），返回奖励数组。

    参数:
        state_eef_position: (T,D)
        action_eef_position: (T,D) 或 None (未使用)
        endpoint: (3,) 或 None
        pos_slice: 位置列范围
    返回:
        (T,) float32 奖励
    """
    return compute_rewards_from_state(state_eef_position, endpoint=endpoint, pos_slice=pos_slice, rpy_slice=rpy_slice, ori_weight=ori_weight)


def compute_rewards_from_h5(h5_path: str | bytes | "os.PathLike[str]") -> np.ndarray:
    """直接读取单个 aligned_joints.h5 计算奖励。

    期望 HDF5 结构包含:
      state/eef/position  (T,12)
      endpoint            (3,)  (若不存在则使用最后一点)
    """
    with h5py.File(h5_path, 'r') as f:
        if 'state/eef/position' not in f:
            raise KeyError('缺少 state/eef/position')
        state_pos = f['state/eef/position'][...]
        endpoint = f['endpoint'][...] if 'endpoint' in f else None
    return offline_reward(state_pos, None, endpoint=endpoint)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='离线奖励计算: progress-based 1 - remaining/total')
    ap.add_argument('--h5', required=True, help='aligned_joints.h5 路径')
    args = ap.parse_args()
    r = compute_rewards_from_h5(args.h5)
    print('rewards shape:', r.shape, 'min/max:', float(r.min()), float(r.max()))

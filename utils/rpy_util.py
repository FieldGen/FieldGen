#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def generate_rpy_trajectory(rpy_start, rpy_end, n):
    """
    生成严格单调递减旋转步长的 RPY 轨迹
    共 n 个点（含起止）
    """
    q0 = R.from_euler('xyz', rpy_start)
    q1 = R.from_euler('xyz', rpy_end)

    # 保证四元数同半球
    if q0.as_quat() @ q1.as_quat() < 0:
        q1 = R.from_quat(-q1.as_quat())

    # 总旋转角（弧度）
    total_angle = (q1 * q0.inv()).magnitude()

    # 缓动时间 0->1，二次缓出（由快到慢）
    t = np.linspace(0, 1, n)
    eased = 1 - (1 - t) ** 2

    # 逐帧旋转角度
    angles = eased * total_angle

    # 用角轴构造每一步的四元数
    axis = (q1 * q0.inv()).as_rotvec()
    if total_angle > 1e-9:
        axis /= total_angle   # 单位轴

    q_traj = [q0]
    for a in angles[1:]:
        q_step = R.from_rotvec(a * axis)
        q_traj.append(q_step * q0)   # 相对于起始帧
    q_traj = R.concatenate(q_traj)

    return q_traj.as_euler('xyz')
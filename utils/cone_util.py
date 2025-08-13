#!/usr/bin/env python3
"""
cone_strict_end.py
交互式 3D：圆锥 + 轨迹 + action 箭头
保证所有轨迹最终**精确**收敛到 O
"""

from ast import main
import numpy as np
import plotly.graph_objects as go

# ---------------- 工具 ----------------
def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n == 0 else v / n

def angle_between(u, v):
    u, v = normalize(u), normalize(v)
    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

# ---------------- 运动模型（保留原逻辑） ----------------
def compute_action_smooth(O, A, B, theta, alpha=1.0, beta=1.0):
    OA, OB = A - O, B - O
    r = np.linalg.norm(OB)
    if r == 0:
        return np.zeros_like(O, dtype=float)

    k_hat, phi = normalize(OA), angle_between(OB, OA)
    theta_r = np.radians(theta)

    # 圆锥外
    if phi > theta_r + 1e-9:
        proj_len = np.dot(OB, k_hat)
        radial_vec = OB - proj_len * k_hat
        radial_dir = normalize(radial_vec) if np.linalg.norm(radial_vec) else normalize(np.cross(k_hat, [1, 0, 0]))
        d_angle = phi - theta_r
        return d_angle * r * (-radial_dir)

    # 圆锥内
    dist_axis = np.linalg.norm(np.cross(OB, k_hat))
    axis_thresh = 0.3
    t = max(0, (axis_thresh - dist_axis) / axis_thresh)
    alpha_eff = alpha * (1 / (1 + np.exp(10 * (t - 0.5))))
    alpha_eff = max(alpha_eff, 0.1)

    dir_along = normalize(np.cross(k_hat, np.cross(k_hat, OB)))
    dir_radial = -k_hat
    return (alpha_eff * dir_along + beta * dir_radial) * r

# ------------ 轨迹生成（步长裁剪 + 强制到位） ------------
def generate_cone_trajectory(start, end, num, theta=30,
                                 dt=0.05,
                                 eps_stop=1e-4):
    traj, acts = [end.copy()], []
    for step in range(num):
        cur = traj[-1]
        r = np.linalg.norm(cur - start)
        if r < eps_stop:                  # 到达
            traj[-1] = start                  # 强制精确到 O
            return np.array(traj), np.array(acts)

        # 方向、速度、裁剪
        A = [start[0], start[1] + 1, start[2]]
        v = compute_action_smooth(start, A, cur, theta)
        v_len = np.linalg.norm(v)
        if v_len == 0:                    # 已在 O
            return np.array(traj), np.array(acts)

        step_len = min(v_len * dt, 0.8 * r)   # 不超 0.8 倍剩余距离
        step_vec = normalize(v) * step_len
        end_step = cur + step_vec
        traj.append(end_step)
        acts.append(step_vec / dt)

    # 若未收敛，也强制终点为 O
    traj.append(O)
    return np.array(traj), np.array(acts)

# ------------ 圆锥网格 ------------
def create_cone_mesh(O, A, theta, height=2.5, n_circle=64, n_gen=50):
    k = normalize(A - O)
    h = np.linspace(0, height, n_gen)
    angles = np.linspace(0, 2 * np.pi, n_circle)
    H, P = np.meshgrid(h, angles)
    R = H * np.tan(np.radians(theta))

    e1 = normalize(np.cross(k, [0, 0, 1] if abs(k[2]) < 0.9 else [1, 0, 0]))
    e2 = np.cross(k, e1)

    X = O[0] + R * np.cos(P) * e1[0] + R * np.sin(P) * e2[0] + H * k[0]
    Y = O[1] + R * np.cos(P) * e1[1] + R * np.sin(P) * e2[1] + H * k[1]
    Z = O[2] + R * np.cos(P) * e1[2] + R * np.sin(P) * e2[2] + H * k[2]
    return X, Y, Z

if __name__ == '__main':
    # ------------ 场景参数 ------------
    O = np.array([0., 0., 0.])
    A = np.array([0., 0., 2.])
    theta = 30.

    B_points = [
        np.array([2., 1., 1.5]),
        np.array([3., 2., 2.]),
        np.array([-1.5, -1., 1.]),
        np.array([2.5, -2., 0.5]),
    ]
    colors = ['crimson', 'lime', 'orange', 'magenta']

    # ------------ 绘图 ------------
    data = []

    for B0, color in zip(B_points, colors):
        traj, acts = generate_cone_trajectory(O, A, B0, theta)
        # 轨迹线
        data.append(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                                mode='lines', line=dict(color=color, width=4),
                                name=f'traj-{color}'))
        # 起点
        data.append(go.Scatter3d(x=[B0[0]], y=[B0[1]], z=[B0[2]],
                                mode='markers', marker=dict(size=6, color=color),
                                name=f'start-{color}'))
        # 稀疏箭头
        mask = np.arange(0, len(acts), max(1, len(acts)//20))
        for i in mask:
            p, v = traj[i], acts[i] * 0.1
            data.append(go.Cone(x=[p[0]], y=[p[1]], z=[p[2]],
                                u=[v[0]], v=[v[1]], w=[v[2]],
                                colorscale=[[0, color], [1, color]],
                                sizeref=0.3, showscale=False, anchor='tail'))

    # 圆锥
    X, Y, Z = create_cone_mesh(O, A, theta)
    data.append(go.Surface(x=X, y=Y, z=Z, opacity=0.25, colorscale='Blues',
                        showscale=False, name='cone'))

    # ------------ 布局 ------------
    fig = go.Figure(data=data)
    xyz = np.vstack([O, A] + B_points)
    margin = 0.5
    mins, maxs = xyz.min(axis=0) - margin, xyz.max(axis=0) + margin
    fig.update_layout(
        scene=dict(xaxis=dict(range=[mins[0], maxs[0]], title='X'),
                yaxis=dict(range=[mins[1], maxs[1]], title='Y'),
                zaxis=dict(range=[mins[2], maxs[2]], title='Z'),
                aspectmode='cube'),
        title='Cone Grasp – Strict End at O',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.write_html("cone_strict_end.html")
    print("已生成交互式网页：cone_strict_end.html")
    fig.show()
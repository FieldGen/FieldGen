#!/usr/bin/env python3
"""
Cone Trajectory – inner cycloid + OA-directed entry
"""

import numpy as np
import plotly.graph_objects as go

# ---------- 小工具 ----------
def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v

# ---------- 圆锥内判断 ----------
def inside_cone(B, O, OA, theta):
    OB = B - O
    r = np.linalg.norm(OB)
    if r == 0:
        return True
    phi = np.arccos(np.clip(np.dot(OB, OA) / r, -1, 1))
    return phi <= np.radians(theta)

# ---------- 直线沿一般轴向 OA 到圆锥面 ----------
def straight_to_cone(B, O, OA, theta, n_rough=400):
    """从点 B 沿圆锥轴向 OA 方向 (单位向量) 前进，求与给定半顶角 theta 的圆锥面交点并生成插值点。

    圆锥定义（顶点 O，轴向 OA，半顶角 theta）：
        设任意点 P 的向量 OP = P - O
        轴向坐标 a = dot(OP, OA)   (要求 a >= 0)
        径向向量 r_vec = OP - a * OA,  r = ||r_vec||
        圆锥面方程: r = tan(theta) * a

    若从 B 沿 OA 方向作直线 B + t * OA (t>=0)，其径向分量保持不变，只需解方程得到 t。
    """
    B = np.asarray(B, dtype=float)
    O = np.asarray(O, dtype=float)
    OA = normalize(OA)

    m = np.tan(np.radians(theta))
    OB = B - O
    a0 = np.dot(OB, OA)                # 轴向投影 (可为负，表示在顶点另一侧)
    r_vec = OB - a0 * OA               # 径向分量
    r0 = np.linalg.norm(r_vec)

    # 已经在圆锥内或在面上：r0 <= m * a0 且 a0 >= 0
    if a0 >= 0 and r0 <= m * a0 + 1e-12:
        hit = B.copy()
    else:
        # 解 r0 = m * (a0 + t)  ->  t = r0/m - a0
        # 若 t < 0 说明 B 朝 -OA 方向才会遇到锥面，此时直接设为 B（退化），避免数值问题
        t = r0 / m - a0
        if t < 0:
            hit = B.copy()
        else:
            hit = B + t * OA

    # 生成非均匀插值点（靠近 hit 更密集）
    u = np.linspace(0, 1, n_rough)
    power = 2.0
    u_nonuniform = u ** power
    line_points = B + u_nonuniform[:, None] * (hit - B)
    return line_points

# ---------- 圆锥内摆线 ----------
def cone_inner_cycloid(B, O, A, n_rough=200):
    """
    Generate a cycloid curve from B to O in the AOB plane.
    B: starting point
    O: end point (cone vertex)
    A: a point on the cone axis, defining the axis direction OA
    n_rough: number of points to sample on the curve
    """
    B = np.asarray(B, dtype=float)
    O = np.asarray(O, dtype=float)
    A = np.asarray(A, dtype=float)

    OA = A - O
    OB = B - O

    # 1. Define the coordinate system of the AOB plane
    # y'-axis is along the cone axis OA
    y_prime_axis = normalize(OA)
    
    # x'-axis is in the AOB plane and perpendicular to the y'-axis
    # It's the component of OB perpendicular to OA
    proj_OB_on_OA = np.dot(OB, y_prime_axis) * y_prime_axis
    x_prime_vec = OB - proj_OB_on_OA
    
    # Handle the case where B is on the axis
    if np.linalg.norm(x_prime_vec) < 1e-6:
        # B is on the axis, the curve is a straight line from B to O
        return np.linspace(B, O, n_rough)
        
    x_prime_axis = normalize(x_prime_vec)

    # 2. Find the coordinates of B in the new (x', y') coordinate system
    # The origin of this system is O
    x_b_prime = np.linalg.norm(x_prime_vec)
    y_b_prime = np.dot(OB, y_prime_axis)

    # 3. Generate a 2D cycloid from (0, 0) to (x_b_prime, y_b_prime)
    # We use one half-arch of a standard cycloid, scaled.
    # Standard half-arch: t from 0 to pi -> x=(t-sin(t)), y=(1-cos(t))
    # This goes from (0,0) to (pi, 2)
    
    # 非均匀采样：越靠近终点O，步幅越小
    # 使用指数函数来创建非均匀的参数分布
    # 参数u从0到1，映射到t从0到pi
    u = np.linspace(0, 1, n_rough)
    
    # 使用指数函数创建非均匀分布，靠近终点(u=1)时步幅更小
    # 可以通过调整power参数来控制非均匀程度
    power = 2.0  # power > 1 使得靠近终点时采样更密集
    u_nonuniform = u ** power
    
    # 将非均匀参数映射到摆线参数t
    t = u_nonuniform * np.pi
    
    # Scale factors to map the half-arch to our target point (x_b_prime, y_b_prime)
    x_prime_curve = (x_b_prime / np.pi) * (t - np.sin(t))
    y_prime_curve = (y_b_prime / 2.0) * (1 - np.cos(t))

    # 4. Transform the 2D curve back to the original 3D coordinate system
    # The curve is a linear combination of the basis vectors x_prime_axis and y_prime_axis,
    # translated by the origin O.
    curve = O + x_prime_curve[:, np.newaxis] * x_prime_axis + y_prime_curve[:, np.newaxis] * y_prime_axis
    
    # The curve is generated from O to B, so we reverse it to go from B to O.
    return curve[::-1]

# ---------- 主轨迹（带标记） ----------
def generate_cone_trajectory(start, end, direct, num, theta=60):
    O  = np.asarray(end, dtype=float)
    OA = normalize(direct)

    # 原始曲线
    if inside_cone(start, O, OA, theta):
        pts = cone_inner_cycloid(start, O, O + OA, n_rough=num)  # 使用 O+OA 作为点 A
        hit_pt = None
    else:
        # 先粗略求交点（两个点即可估算线段长度占比）
        line_pts = straight_to_cone(start, O, OA, theta, n_rough=2)
        cycl_pts = cone_inner_cycloid(line_pts[-1], O, O + OA, n_rough=400)
        pts_tmp = np.vstack([line_pts[:-1], cycl_pts])
        hit_pt = line_pts[-1]

        # 估算比例以重新分配采样数量
        line_length = np.linalg.norm(line_pts[-1] - line_pts[0])
        point_diffs = np.diff(cycl_pts, axis=0)
        segment_lengths = np.linalg.norm(point_diffs, axis=1)
        cycl_length = np.sum(segment_lengths)

        line_num = max(int(num * line_length / (line_length + cycl_length)), 2)
        cycl_num = max(1, num - line_num + 1)

        line_pts = straight_to_cone(start, O, OA, theta, n_rough=line_num)
        cycl_pts = cone_inner_cycloid(line_pts[-1], O, O + OA, n_rough=cycl_num)
        pts = np.vstack([line_pts[:-1], cycl_pts])

    return pts

# ---------------- 绘图（含标记） ----------------
if __name__ == "__main__":
    B = np.array([2, 3, 1])
    O = np.array([0, 0, 0])
    axis_dir = np.array([0, -1, 0])  # 示例：轴向向下
    traj, hit = generate_cone_trajectory(B, O, axis_dir, num=200, theta=30)

    fig = go.Figure()

    # 整条轨迹
    fig.add_trace(go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
        mode='lines+markers',
        line=dict(color='royalblue', width=5),
        name='B→O curve'))

    # 起点（红色球）
    fig.add_trace(go.Scatter3d(
        x=[B[0]], y=[B[1]], z=[B[2]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='start'))

    # 终点（绿色球）
    fig.add_trace(go.Scatter3d(
        x=[O[0]], y=[O[1]], z=[O[2]],
        mode='markers',
        marker=dict(color='green', size=10),
        name='end'))

    # hit 点（如果存在，黄色球）
    if hit is not None:
        fig.add_trace(go.Scatter3d(
            x=[hit[0]], y=[hit[1]], z=[hit[2]],
            mode='markers',
            marker=dict(color='yellow', size=10),
            name='hit'))

    # 圆锥面
    h = 4
    t, a = np.linspace(0, h, 50), np.linspace(0, 2*np.pi, 64)
    T, A = np.meshgrid(t, a)
    R = T * np.tan(np.radians(30))
    fig.add_trace(go.Surface(x=R*np.cos(A), y=-T, z=R*np.sin(A),
                             opacity=0.2, colorscale='Blues', showscale=False))

    fig.update_layout(scene=dict(aspectmode='cube'),
                      title='Cone Trajectory with Start / End / Hit Markers')
    fig.write_html("cone_trajectory.html")
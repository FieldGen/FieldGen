import numpy as np
import math

def point_to_plane_distance(point, plane_normal, plane_point):
    # 点到平面距离
    return np.abs(np.dot(plane_normal, point - plane_point)) / np.linalg.norm(plane_normal)

def bezier_curve(points, num):
    # 三次方贝塞尔曲线
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 3))
    for i in range(n + 1):
        binom = math.comb(n, i)
        curve += binom * ((1 - t) ** (n - i))[:, None] * (t ** i)[:, None] * points[i]
    return curve

def generate_bezier_trajectory(start, end, num=200, safe_plane_y=-0.2):
    # 判断起点在安全平面左还是右
    if start[1] > safe_plane_y:
        # 左侧，5点
        P0 = np.array([start[0], safe_plane_y, start[2]])
        P1 = np.array([start[0], safe_plane_y, end[2]])
        P2 = np.array([end[0], safe_plane_y, end[2]])
        control_points = [start, P0, P1, P2, end]
    else:
        # 右侧，4点
        P0 = np.array([start[0], start[1], end[2]])
        P1 = np.array([end[0], start[1], end[2]])
        control_points = [start, P0, P1, end]
    # 生成贝塞尔曲线
    curve = bezier_curve(control_points, num)
    # 采样m个最近点（只用输入采样点）
    return curve
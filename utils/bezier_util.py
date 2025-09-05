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

def generate_bezier_trajectory(start, end, direct=None, num=2000):
    """生成一条从 start 到 end 的贝塞尔轨迹。

    新逻辑：给定 end 和方向向量 direct 构成一条射线 L: end + t * direct。
    计算 start 到该直线的垂足(投影点) P，使用 [start, P, end] 三个控制点
    构造二次贝塞尔曲线。

    参数:
        start (array-like): 起点 3D
        end (array-like): 终点 3D
        direct (array-like|None): 与 end 组成射线的方向向量；若为 None，默认使用 end-start
        num (int): 采样点数量
        vertical (bool): 为兼容旧接口保留（已无实际作用）
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if direct is None:
        direct = end - start
    direct = np.asarray(direct, dtype=float)

    norm = np.linalg.norm(direct)
    if norm < 1e-9:
        # 如果方向向量长度近似为0，则退化：取中点作为控制点
        proj_point = (start + end) / 2.0
    else:
        u = direct / norm
        # 向量从 end 指向 start
        v = start - end
        # 标量投影长度（可正可负，表示在射线方向上的位置）
        t = np.dot(v, u)
        proj_point = end + t * u

    control_points = [start, proj_point, end]
    curve = bezier_curve(control_points, num)
    return curve
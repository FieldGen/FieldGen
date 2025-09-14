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

def generate_bezier_trajectory(start, end, direct=None, num=2000, nonuniform=True, power=2.0):
    """生成一条从 start 到 end 的贝塞尔轨迹，并支持非均匀采样。

    逻辑：给定 end 和方向向量 direct 构成射线 L: end + t * direct。
    计算 start 到该直线的垂足(投影点) P，使用 [start, P, end] 作为控制点构造二次贝塞尔曲线。

    若 ``nonuniform`` 为 True，则对参数 t 采用加权分布（u ** power），实现靠近终点更密集的采样，
    类似 `cone_util.cone_inner_cycloid` 中的做法。

    参数:
        start (array-like): 起点 3D
        end (array-like): 终点 3D
        direct (array-like|None): 与 end 组成射线的方向向量；若为 None，默认使用 end-start
        num (int): 采样点数量
        nonuniform (bool): 是否启用非均匀参数采样
        power (float): 非均匀采样幂指数 (>1 终点更密集, <1 起点更密集, =1 均匀)
    返回:
        ndarray shape (num,3): 轨迹点
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
    # 基础均匀贝塞尔采样（参数 s 均匀）
    if not nonuniform:
        return bezier_curve(control_points, num)

    # 非均匀：先生成高分辨率均匀曲线，再根据加权参数重采样
    # 为保持精度，内部先用 >= num 的分辨率（这里直接用 num）
    base_pts = bezier_curve(control_points, num)

    # 计算沿曲线的弧长参数（累积距离 -> 归一化）
    diffs = np.diff(base_pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    if arc[-1] < 1e-12:
        return base_pts  # 退化成一个点
    arc /= arc[-1]

    # 目标参数分布：u in [0,1]
    # 为与 cone_inner_cycloid 一致：终点更密集。
    # cone 里是 u**power 但曲线方向最后被反转；这里方向不反转，因此使用 1-(1-u)**power 达到末端密集。
    u = np.linspace(0, 1, num)
    if power == 1.0:
        w = u
    else:
        # power>1 终点更密集；power<1 起点更密集；=1 均匀
        w = 1.0 - (1.0 - u) ** power

    # 在弧长参数空间插值取得对应点
    # 对每个维度做一次线性插值
    resampled = np.empty((num, 3), dtype=float)
    for k in range(3):
        resampled[:, k] = np.interp(w, arc, base_pts[:, k])
    return resampled
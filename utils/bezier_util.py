import numpy as np
import math

def point_to_plane_distance(point, plane_normal, plane_point):
    # 点到平面距离
    return np.abs(np.dot(plane_normal, point - plane_point)) / np.linalg.norm(plane_normal)

def bezier_curve(points, num=100):
    # 三次贝塞尔曲线
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 3))
    for i in range(n + 1):
        binom = math.comb(n, i)
        curve += binom * ((1 - t) ** (n - i))[:, None] * (t ** i)[:, None] * points[i]
    return curve

def find_nearest_points(curve, sample_points, m):
    # curve: (N,3), sample_points: [{'img':..., 'state':..., 'xyz':np.array([x,y,z])}, ...]
    sample_xyz = np.array([p['xyz'] for p in sample_points])
    used = set()
    result = []
    for pt in np.linspace(0, len(curve)-1, m).astype(int):
        dists = np.linalg.norm(sample_xyz - curve[pt], axis=1)
        # 排除已用过的点
        for idx in np.argsort(dists):
            if idx not in used:
                used.add(idx)
                result.append(sample_points[idx])
                break
        if len(result) == m:
            break
    # 如果还不够，补足
    if len(result) < m:
        for idx in range(len(sample_points)):
            if idx not in used:
                result.append(sample_points[idx])
                if len(result) == m:
                    break
    return result

def generate_bezier_trajectory(start, end, sample_points, m, safe_plane_x=0):
    # start, end: np.array([x, y, z])
    # 判断起点在安全平面左还是右
    if start[0] < safe_plane_x:
        # 左侧，5点
        # P0: 起点垂直于安全平面的点
        P0 = np.array([safe_plane_x, start[1], start[2]])
        # P1: P0对抓取点水平面的垂直点
        P1 = np.array([safe_plane_x, end[1], start[2]])
        # P2: P1垂直抓取点垂直安全平面的那条射线l的垂直点
        # 射线l: (x=safe_plane_x, y=end[1], z)
        P2 = np.array([safe_plane_x, end[1], end[2]])
        control_points = [start, P0, P1, P2, end]
    else:
        # 右侧，4点
        # P0: 起点对抓取点水平面的垂直点
        P0 = np.array([start[0], end[1], start[2]])
        # P1: P0垂直抓取点垂直安全平面的那条射线l的垂直点
        # 射线l: (x, y=end[1], z=end[2])
        P1 = np.array([start[0], end[1], end[2]])
        control_points = [start, P0, P1, end]
    # 生成贝塞尔曲线
    curve = bezier_curve(control_points, num=200)
    # 采样m个最近点
    traj = find_nearest_points(curve, sample_points, m)
    # 输出
    grasp_point = end
    safe_plane_dist = point_to_plane_distance(end, np.array([1,0,0]), np.array([safe_plane_x,0,0]))
    return traj, grasp_point, safe_plane_dist

# 用法示例
if __name__ == "__main__":
    start = np.array([-0.2, 0.1, 0.3])
    end = np.array([0.3, 0.2, 0.5])
    sample_points = [
        {'img': None, 'state': None, 'xyz': np.array([x, y, z])}
        for x, y, z in np.random.rand(100, 3)
    ]
    m = 10
    traj, grasp, dist = generate_bezier_trajectory(start, end, sample_points, m)
    print("轨迹点数量:", len(traj))
    print("抓取点坐标:", grasp)
    print("安全平面距离:", dist)
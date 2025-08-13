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

def generate_bezier_trajectory(start, end, num=200, safe_plane_x=0):
    # 判断起点在安全平面左还是右
    if start[0] < safe_plane_x:
        # 左侧，5点
        P0 = np.array([safe_plane_x, start[1], start[2]])
        P1 = np.array([safe_plane_x, end[1], start[2]])
        P2 = np.array([safe_plane_x, end[1], end[2]])
        control_points = [start, P0, P1, P2, end]
    else:
        # 右侧，4点
        P0 = np.array([start[0], end[1], start[2]])
        P1 = np.array([start[0], end[1], end[2]])
        control_points = [start, P0, P1, end]
    # 生成贝塞尔曲线
    curve = bezier_curve(control_points, num)
    # 采样m个最近点（只用输入采样点）
    return curve

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
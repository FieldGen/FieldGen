import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_cone_trajectory():
    rpy1 = np.radians([10, 20, 30])   # 目标姿态
    rpy2 = np.radians([5, -15, 45])   # 当前姿态

    # 转成四元数
    q1 = R.from_euler('xyz', rpy1, degrees=False)
    q2 = R.from_euler('xyz', rpy2, degrees=False)

    # 相对旋转 q_rel = q1 * inv(q2)  →  把坐标系2转到坐标系1
    q_rel = q1 * q2.inv()

    # 再转回欧拉角（ZYX）
    delta_rpy = q_rel.as_euler('xyz', degrees=False)  # 弧度
    print(np.degrees(delta_rpy))  # 结果才是“姿态误差”
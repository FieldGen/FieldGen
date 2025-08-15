import sys
from pathlib import Path
# 把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_rpy_trajectory

def step_angle(traj):
    """相邻帧之间的真实旋转角（度）"""
    d = []
    for q1, q2 in zip(R.from_euler('xyz', traj[:-1]),
                      R.from_euler('xyz', traj[1:])):
        d.append((q2 * q1.inv()).magnitude() * 180 / np.pi)
    return np.array(d)


# ----------------- 演示 -----------------
if __name__ == '__main__':
    deg = np.pi / 180
    start = np.array([0, 0, 0]) * deg
    end   = np.array([90, -45, 180]) * deg
    print(start)
    print(end)
    n = 100
    traj = generate_rpy_trajectory(start, end, n)
    print(traj)
    delta = step_angle(traj)

    plt.plot(delta, label='delta angle per step (deg)')
    plt.xlabel('step')
    plt.ylabel('Δθ')
    plt.title('Rotation step strictly decreases')
    plt.grid(); plt.legend()
    plt.savefig('./rpy_traj.jpg')
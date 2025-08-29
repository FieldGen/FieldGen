import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path

# 假设的三个关键位姿 [x, y, z, qx, qy, qz, qw]
poses = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 初始位姿
    [1.0, 1.0, 0.5, 0.0, 0.0, 0.707, 0.707],  # 中间位姿
    [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]   # 结束位姿
])

# 提取位置和四元数
positions = poses[:, :3]
quaternions = poses[:, 3:]

# 时间戳（归一化）
times = np.array([0.0, 0.5, 1.0])

# 1. 位置插值：使用三次样条
pos_spline = CubicSpline(times, positions, bc_type='clamped')  # C2连续性

# 2. 姿态插值：使用Slerp
rotations = R.from_quat(quaternions)
slerp = Slerp(times, rotations)

# 3. 生成插值轨迹
num_points = 100
t_interp = np.linspace(0.0, 1.0, num_points)

# 插值位置
interp_positions = pos_spline(t_interp)

# 插值姿态
interp_rotations = slerp(t_interp)
interp_quaternions = interp_rotations.as_quat()

# 组合成完整的位姿轨迹
interp_poses = np.hstack([interp_positions, interp_quaternions])

# 可视化位置轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(interp_positions[:, 0], interp_positions[:, 1], interp_positions[:, 2], label='Interpolated Path')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='red', label='Key Poses')

# ===== 添加RPY方向向量显示（机体系三个主轴） =====
rot_mats = interp_rotations.as_matrix()  # (N,3,3)
# 机体系x,y,z轴在世界系中的方向向量
axes_dirs = {
    'x': (rot_mats[:, :, 0], 'tab:green'),
    'y': (rot_mats[:, :, 1], 'tab:orange'),
    'z': (rot_mats[:, :, 2], 'tab:blue'),
}
step = max(1, len(interp_positions)//15)  # 采样减少视觉杂乱
scale_len = np.linalg.norm(positions[-1]-positions[0]) * 0.07 + 1e-9
first_labels_done = set()
for axis_name, (dirs, color) in axes_dirs.items():
    p = interp_positions[::step]
    d = dirs[::step]
    # 归一化后统一缩放
    d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True)+1e-9) * scale_len
    label = f'body {axis_name}-axis' if not first_labels_done else None
    ax.quiver(p[:,0], p[:,1], p[:,2], d_norm[:,0], d_norm[:,1], d_norm[:,2], length=1.0, color=color, normalize=False, linewidth=0.8, label=label)
    first_labels_done.add(axis_name)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
# 保存图片而不是直接显示
out_dir = Path(__file__).resolve().parent.parent / 'figs'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'interp_path.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {out_path}")

# 如仍需交互查看，可取消下面注释
# plt.show()
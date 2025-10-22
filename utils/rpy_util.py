#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def generate_rpy_trajectory(rpy_start, rpy_end, n):
    """Generate n-step RPY path with strictly decreasing incremental rotation (ease-out)."""
    q0 = R.from_euler('xyz', rpy_start)
    q1 = R.from_euler('xyz', rpy_end)

    # Same hemisphere to ensure shortest path
    if q0.as_quat() @ q1.as_quat() < 0:
        q1 = R.from_quat(-q1.as_quat())

    # Total rotation angle (radians)
    total_angle = (q1 * q0.inv()).magnitude()

    # Quadratic ease-out in normalized time
    t = np.linspace(0, 1, n)
    eased = 1 - (1 - t) ** 2

    # Scaled cumulative angles
    angles = eased * total_angle

    # Axis-angle incremental composition
    axis = (q1 * q0.inv()).as_rotvec()
    if total_angle > 1e-9:
        axis /= total_angle

    q_traj = [q0]
    for a in angles[1:]:
        q_step = R.from_rotvec(a * axis)
        q_traj.append(q_step * q0)
    q_traj = R.concatenate(q_traj)

    return q_traj.as_euler('xyz')
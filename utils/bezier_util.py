import numpy as np
import math

# Removed unused helper to keep surface minimal.

def bezier_curve(points, num):
    """Evaluate an n-th order Bézier curve for given control points (Cartesian 3D)."""
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 3))
    for i in range(n + 1):
        binom = math.comb(n, i)
        curve += binom * ((1 - t) ** (n - i))[:, None] * (t ** i)[:, None] * points[i]
    return curve

def generate_bezier_trajectory(start, end, direct=None, num=2000, nonuniform=True, power=2.0):
    """Generate a quadratic Bézier trajectory with optional nonuniform (arc-length based) sampling.

    The intermediate control point is the orthogonal projection of start onto the ray defined
    by ``end`` and ``direct`` (defaults to end-start). A power-law warping of cumulative arc-length
    biases sampling density toward the terminal segment (power > 1) or the initial segment (<1).

    Parameters
    ----------
    start : array-like (3,)
    end : array-like (3,)
    direct : array-like (3,), optional
    num : int
    nonuniform : bool
    power : float

    Returns
    -------
    np.ndarray (num,3)
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if direct is None:
        direct = end - start
    direct = np.asarray(direct, dtype=float)

    norm = np.linalg.norm(direct)
    if norm < 1e-9:
        # Degenerate: use midpoint as control point
        proj_point = (start + end) / 2.0
    else:
        u = direct / norm
        v = start - end  # vector from end to start
        t = np.dot(v, u)  # scalar projection length
        proj_point = end + t * u

    control_points = [start, proj_point, end]
    # Uniform parameter sampling
    if not nonuniform:
        return bezier_curve(control_points, num)

    # Nonuniform: generate uniform curve then resample by inverse CDF over arc-length.
    base_pts = bezier_curve(control_points, num)

    # Cumulative arc-length normalization
    diffs = np.diff(base_pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    if arc[-1] < 1e-12:
        return base_pts  # 退化成一个点
    arc /= arc[-1]

    # Power-law warping for density shaping
    u = np.linspace(0, 1, num)
    if power == 1.0:
        w = u
    else:
    # power>1 terminal-heavy; <1 start-heavy
        w = 1.0 - (1.0 - u) ** power

    # Linear interpolation in arc-length parameter space
    resampled = np.empty((num, 3), dtype=float)
    for k in range(3):
        resampled[:, k] = np.interp(w, arc, base_pts[:, k])
    return resampled
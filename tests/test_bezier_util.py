
import unittest
import numpy as np
import matplotlib.pyplot as plt
from utils.bezier_util import generate_bezier_trajectory


class TestBezierUtil(unittest.TestCase):
    def setUp(self):
        self.start = np.array([-0.5, -0.5, -0.5])
        self.end = np.array([0.5, 0.5, 0.5])
        np.random.seed(42)
        n = 100000
        self.m = 100
        self.sample_points = [
            {'img': None, 'state': None, 'xyz': np.array([x, y, z])}
            for x, y, z in np.random.rand(n, 3) * np.array([2, 2, 2]) - np.array([1, 1, 1])
        ]

    def test_visualization(self):
        # 可视化测试（不做断言，仅生成图片）
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        curve = generate_bezier_trajectory(self.start, self.end)
        ax.plot(curve[:,0], curve[:,1], curve[:,2], 'b--', label='bezier curve', linewidth=1)
        y = np.linspace(-0.5, 0.5, 10)
        z = np.linspace(-0.5, 0.5, 10)
        Y, Z = np.meshgrid(y, z)
        X = np.zeros_like(Y) + 0
        ax.plot_surface(X, Y, Z, alpha=0.2, color='g')
        ax.scatter(self.start[0], self.start[1], self.start[2], c='k', s=80, marker='o', label='S')
        ax.scatter(self.end[0], self.end[1], self.end[2], c='m', s=80, marker='*', label='T')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Bezier Trajectory Visualization')
        plt.savefig("bezier_plot.png")

if __name__ == "__main__":
    unittest.main()
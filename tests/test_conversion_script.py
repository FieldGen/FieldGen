import unittest
from pathlib import Path
import h5py
import numpy as np
import subprocess
import sys

SCRIPT_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'convert_aligned_to_rl_format.py'
INPUT_FILE = Path(__file__).resolve().parent.parent / 'aligned_joints.h5'
OUTPUT_FILE = Path(__file__).resolve().parent.parent / 'converted_dataset.hdf5'

class TestConversionScript(unittest.TestCase):
    def setUp(self):
        if not INPUT_FILE.exists():
            self.skipTest(f"缺少输入文件 {INPUT_FILE}, 跳过转换测试")
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

    def test_run_conversion(self):
        cmd = [sys.executable, str(SCRIPT_PATH),
               '--input', str(INPUT_FILE),
               '--output', str(OUTPUT_FILE),
               '--env-name', 'DummyEnv-v0']
        subprocess.check_call(cmd)
        self.assertTrue(OUTPUT_FILE.exists(), '输出文件未生成')
        with h5py.File(OUTPUT_FILE, 'r') as f:
            self.assertIn('total', f.attrs)
            self.assertEqual(f.attrs['total'], 1)
            self.assertIn('env_args', f.attrs)
            self.assertIn('demo_0', f)
            demo = f['demo_0']
            for key in ['obs', 'next_obs', 'actions', 'rewards', 'dones']:
                self.assertIn(key, demo)
            obs = demo['obs']['flat'][...]
            next_obs = demo['next_obs']['flat'][...]
            actions = demo['actions'][...]
            rewards = demo['rewards'][...]
            dones = demo['dones'][...]
            T = obs.shape[0]
            # 基本形状关系
            self.assertEqual(obs.shape, next_obs.shape)
            self.assertEqual(obs.shape, actions.shape)
            self.assertEqual(rewards.shape, (T,))
            self.assertEqual(dones.shape, (T,))
            # next_obs 最后一帧 == obs 最后一帧
            np.testing.assert_allclose(next_obs[-1], obs[-1])
            # dones 最后一帧为 1，其余为 0
            self.assertEqual(dones[-1], 1)
            self.assertTrue(np.all(dones[:-1] == 0))

if __name__ == '__main__':
    unittest.main()

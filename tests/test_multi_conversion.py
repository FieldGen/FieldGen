import unittest
from pathlib import Path
import h5py
import numpy as np
import subprocess, sys, shutil, tempfile

SCRIPT_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'convert_aligned_to_rl_format.py'

class TestMultiEpisodeConversion(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='multi_ep_'))
        # 创建两个 episode 目录
        for i in range(2):
            ep = self.tmpdir / f'episode{i}'
            ep.mkdir(parents=True, exist_ok=True)
            h5_path = ep / 'aligned_joints.h5'
            T = 5 + i  # 不同长度
            with h5py.File(h5_path, 'w') as f:
                # action/eef/position (T,12), action/effector/position (T,2)
                g_action = f.create_group('action')
                g_action_eef = g_action.create_group('eef')
                g_action_eff = g_action.create_group('effector')
                g_action_eef.create_dataset('position', data=np.random.randn(T,12).astype('f4'))
                g_action_eff.create_dataset('position', data=np.random.randn(T,2).astype('f4'))
                g_state = f.create_group('state')
                g_state_eef = g_state.create_group('eef')
                g_state_eff = g_state.create_group('effector')
                g_state_eef.create_dataset('position', data=np.random.randn(T,12).astype('f4'))
                g_state_eff.create_dataset('position', data=np.random.randn(T,2).astype('f4'))
        self.output = self.tmpdir / 'out.hdf5'

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_multi(self):
        cmd = [sys.executable, str(SCRIPT_PATH),
               '--input-dir', str(self.tmpdir),
               '--output', str(self.output),
               '--env-name', 'DummyEnvDir-v0']
        subprocess.check_call(cmd)
        self.assertTrue(self.output.exists())
        with h5py.File(self.output, 'r') as f:
            self.assertEqual(f.attrs['total'], 2)
            self.assertIn('demo_0', f)
            self.assertIn('demo_1', f)
            for di in range(2):
                demo = f[f'demo_{di}']
                obs = demo['obs']['flat'][...]
                next_obs = demo['next_obs']['flat'][...]
                self.assertEqual(obs.shape, next_obs.shape)
                # next_obs 最后一帧复制
                self.assertTrue(np.allclose(next_obs[-1], obs[-1]))
                dones = demo['dones'][...]
                self.assertEqual(dones[-1], 1)

if __name__ == '__main__':
    unittest.main()

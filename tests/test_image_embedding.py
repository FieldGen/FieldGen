import unittest
from pathlib import Path
import h5py, sys, subprocess, shutil, tempfile
import numpy as np
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'convert_aligned_to_rl_format.py'

class TestImageEmbedding(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='img_ep_'))
        ep = self.tmpdir / 'episode0'
        # 创建多个帧目录 camera/0, camera/1, camera/2
        for fid in range(3):
            (ep / 'camera' / str(fid)).mkdir(parents=True, exist_ok=True)
        # 写入简单 h5
        h5_path = ep / 'aligned_joints.h5'
        import h5py, numpy as np
        T = 4
        with h5py.File(h5_path, 'w') as f:
            g_action = f.create_group('action'); g_state = f.create_group('state')
            for g in [g_action, g_state]:
                ge = g.create_group('eef'); gf = g.create_group('effector')
                ge.create_dataset('position', data=np.random.randn(T,12).astype('f4'))
                gf.create_dataset('position', data=np.random.randn(T,2).astype('f4'))
        # 为每帧写入三张假图片 (不同颜色编码帧索引)
        base_views = [('hand_left.jpg',(255,0,0)), ('hand_right.jpg',(0,255,0)), ('head.jpg',(0,0,255))]
        for fid in range(3):
            for name, base_color in base_views:
                # 颜色稍做变化
                color = tuple(min(255, c + fid*10) for c in base_color)
                img = Image.new('RGB', (16,16), color)
                img.save(ep / 'camera' / str(fid) / name)
        self.output = self.tmpdir / 'out.hdf5'

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_images_present(self):
        cmd = [sys.executable, str(SCRIPT_PATH), '--input-dir', str(self.tmpdir), '--output', str(self.output), '--env-name', 'ImgEnv-v0']
        subprocess.check_call(cmd)
        with h5py.File(self.output, 'r') as f:
            obs = f['demo_0']['obs']
            for ds_name in ['hand_left', 'hand_right', 'head']:
                self.assertIn(ds_name, obs)
                arr = obs[ds_name][...]
                # 3 帧, 每帧 16x16x3
                self.assertEqual(arr.shape, (3,16,16,3))
                self.assertEqual(arr.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()

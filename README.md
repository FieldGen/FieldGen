

# FieldGen-Grasp

仅使用一些点采集(img, state)的基础上，通过不同方法来生成高质量轨迹用于模型训练。

## 目录
- [简介](#简介)
- [安装](#安装)
- [使用方法](#使用方法)
- [测试](#测试)
- [贡献](#贡献)
- [许可证](#许可证)

## 简介


本项目包含如下主要功能：
- 贝塞尔曲线轨迹生成
- 锥体轨迹生成
- 欧拉角（RPY）轨迹生成与三维可视化
- 采集点与轨迹的 2D/3D 绘制
- 支持 YAML 配置、HDF5 数据处理


## 安装

建议使用 Python 3.10 及以上版本。

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourname/fieldgen-grasp.git
   cd fieldgen-grasp
   ```
2. （可选）创建虚拟环境（需已安装 Python 3.10）：
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

   主要依赖包：
   - numpy
   - matplotlib
   - plotly
   - h5py
   - Pillow
   - tqdm
   - scipy
   - pyyaml

## 使用方法



示例：
```python
from utils.bezier_util import generate_bezier_trajectory
from utils.cone_util import generate_cone_trajectory
from utils.rpy_util import generate_rpy_trajectory
from utils.visualize_points import visualize_curve_with_rpy
```

### 数据集格式转换 (aligned_joints.h5 -> RL episodic)

新增脚本：`scripts/convert_aligned_to_rl_format.py`

输入 HDF5 假定结构：
```
aligned_joints.h5
├── timestamps                 # (T,)
├── action/
│   ├── eef/position           # (T, 12)
│   └── effector/position      # (T, 2)  (夹爪动作，已缩放 gripper/90)
└── state/
      ├── eef/position           # (T, 12)
      └── effector/position      # (T, 2)  (原始 gripper 值)
```

输出示例：
```
dataset.hdf5
   attrs:
      total: 1
      env_args: '{"env_name": "DummyEnv-v0", ...}'
   demo_0/
      attrs:
         num_samples: T
      obs/flat        (T, 14)
      next_obs/flat   (T, 14)  # 末步复制
      actions         (T, 14)
      rewards         (T,)     # 默认全 0，可配置
      dones           (T,)     # 最后一帧=1
```

运行示例（单文件）：
```bash
python scripts/convert_aligned_to_rl_format.py \
   --input aligned_joints.h5 \
   --output dataset.hdf5 \
   --env-name DummyEnv-v0 \
   --env-args-json '{"max_steps":200}'
```

多 episode 目录（目录包含 episode*/aligned_joints.h5）：
```bash
python scripts/convert_aligned_to_rl_format.py \
   --input-dir /path/to/processed.mainexp2.tele \
   --output dataset_multi.hdf5 \
   --env-name DummyEnv-v0 \
   --shuffle --seed 42 --max-episodes 100
```

可选参数：
* `--model-file path.xml` 写入 demo attrs
* `--obs-components eef/position effector/position` 自定义观测拼接组件（state 下）
* `--action-components eef/position effector/position` 自定义动作组件（action 下）
* `--sparse-final-reward 1.0` 末步附加奖励
* `--reward-per-step 0.0` 每步基础奖励
* `--truncate 500` 截断时间长度
* `--compression gzip` HDF5 数据集压缩
* `--copy-gripper-from-state` 用 state 中 gripper 值覆盖 action 中对应部分
* `--input-dir` 目录模式，自动发现 episode*/aligned_joints.h5
* `--shuffle` 目录模式下随机顺序
* `--max-episodes N` 限制最多转换数量
* `--seed` 与 shuffle 配合

默认 obs 与 action 维度均为 14 (12 关节 + 2 夹爪)。

若 episode 目录存在相机帧目录结构：`camera/<frame_index>/<view>.jpg`（其中 `<view>` ∈ {hand_left.jpg, hand_right.jpg, head.jpg} 且 `<frame_index>` 为数字），脚本会收集所有帧按索引排序，并将同一 view 的序列堆叠为形如 `(F,H,W,3)` 的 `uint8` 数据集写入 `demo_i/obs/`，数据集名称去掉 `.jpg` 扩展（例如 `hand_left`）。如果帧间分辨率不同，会裁剪到所有帧的最小中心区域。`next_obs` 不包含图片。

测试：
```bash
python -m unittest tests.test_conversion_script
```

## 测试



运行全部测试用例：
```bash
python -m unittest discover tests
```

## 贡献

欢迎提交 issue 或 pull request。

## 许可证



本项目采用 MIT License。

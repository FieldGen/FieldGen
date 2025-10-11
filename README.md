

# FieldGen-Grasp

仅使用一些点采集(img, state)的基础上，通过不同方法来生成高质量轨迹用于模型训练。

## 目录
- [简介](#简介)
<div align="center">

# FieldGen-Grasp

基于稀疏采样点 (images + state/eef) 的数据增强 / 轨迹生成工具集：从少量末端执行器候选位置出发，自动生成平滑、可控、长度归一的抓取或接近轨迹，并支持随机扰动与奖励估计，用于下游模仿学习 / 强化学习数据扩展。

</div>

## 🔍 背景概述
在真实机器人采集中，往往只能获得少量关键“兴趣点”(例如人工挑选 / 采样的若干末端姿态)。本项目通过几何启发（贝塞尔 + 圆锥内摆线）与可调采样密度，自动补全形成高质量连续轨迹，同时生成配套的姿态(RPY)、夹爪开合、图像帧及统计信息，降低人工标注或大量真实 rollouts 成本。

## ✨ 主要特性
* Bezier / Cone 两类轨迹生成器（可扩展）
* RPY 缓出式旋转轨迹（严格单调减小角速度）
* 非均匀采样：靠近终点区域更高密度，利于精细操作学习
* 长度归一 / 截断 / 补齐逻辑，适配固定 chunk_size 的训练流水线
* 支持随机终点扰动 + 简单 reward 估计（`generate_random.py`）
* 多任务批量处理：统一统计（曲线长度、扩展/截断计数、图像缺失等）
* HDF5 输出结构清晰：action/state/eef/effector + timestamps (+ reward)
* 点云/轨迹/密度 分布 Plotly 交互可视化与 Matplotlib 3D 帧方向箭头显示
* YAML 配置一键切换曲线类型、输出路径、任务列表及随机半径等参数

## 📦 安装
建议 Python >= 3.10。

```bash
git clone https://github.com/yourname/fieldgen-grasp.git
cd fieldgen-grasp
python3.10 -m venv .venv  # 可选
source .venv/bin/activate
pip install -r requirements.txt
```

核心依赖：numpy / scipy / matplotlib / plotly / h5py / Pillow / tqdm / pyyaml

## 🗂 数据与输出格式
输入（每个原始任务目录示例）
```
<task_root>/
   sample_points.h5
   camera/
      0/hand_left.jpg ...
      1/hand_left.jpg ...   # 可选多帧
```
`sample_points.h5` 需包含：
* `state/eef/position` : (N, >=12) 行为/状态混合向量，脚本使用列 6:9 (xyz_start), 9:12 (rpy_start)
* `endpoint` : (>=12,) 终点描述，使用 6:9 (xyz_end), 9:12 (rpy_end)

生成输出（`output_path/episodeK/`）
```
episodeK/
   aligned_joints.h5
   camera/0/<view>.jpg  # 复制来源帧（全部放到时间索引 0 下）
   curve_visualization.png  # 可选（若启用）
```
`aligned_joints.h5` 结构：
```
timestamps : (T,)
action/eef/position : (T, 12)  [左臂(6) + 右/生成曲线(6)]
state/eef/position  : (T, 12)
action/effector/position : (T,2)  # 归一化(除以90)
state/effector/position  : (T,2)  # 原始角度
reward/value (T,) 或 标量    # 仅 random 版本脚本
```
说明：`combined_data = [left_xyz(3), left_rpy(3), right_xyz(3), right_rpy(3)]` → 12 维。

## ⚙️ 配置说明 (`config/config.yaml`)
关键字段：
* `generate.curve_type`: `bezier` | `cone`
* `generate.output_path`: 轨迹写出根目录
* `generate.chunk_size`: 归一化长度 T（不足补齐，超出截断）
* `generate.beta`: 控制点数密度（影响二次采样点数）
* `generate.endpoint_random_radius`: 随机扰动终点球半径（仅 `generate_random.py`）
* `tasks.*.path`: 各任务源数据目录
* `tasks.*.max_trajectories`: 限制每任务生成 episode 数（null = 全部）

## 🚀 快速开始
1. 准备多个任务目录并放置 `sample_points.h5` 与可选 `camera/*/*.jpg`
2. 修改 `config/config.yaml`
3. 运行（确定当前目录为仓库根）：
```bash
python generate.py          # 生成确定终点轨迹
python generate_random.py   # 生成随机终点 + reward
```
4. 查看统计输出与生成的 `aligned_joints.h5`
5. (可选) 可视化点分布：
```bash
python utils/visualize_points.py
```

## 🧠 核心脚本简介
| 脚本/模块 | 功能 |
|-----------|------|
| `generate.py` | 批量读取任务点集，生成统一长度轨迹与姿态 |
| `generate_random.py` | 在终点附近采样随机终点并估计 reward |
| `utils/bezier_util.py` | 二次 Bezier + 非均匀弧长重采样 |
| `utils/cone_util.py` | 圆锥进入 + 内摆线轨迹，终点密集 |
| `utils/rpy_util.py` | RPY 缓出插值（角速度单调下降） |
| `utils/visualize_points.py` | 点云 3D / 三视图密度 HTML 输出 |

## 🧪 轨迹生成示例 (API)
```python
from utils.bezier_util import generate_bezier_trajectory
from utils.cone_util import generate_cone_trajectory
from utils.rpy_util import generate_rpy_trajectory

start = [0.1, 0.2, 0.3]
end   = [0.0, 0.0, 0.5]
direct = [0,0,-1]  # 或通过 get_direct(end, rpy_end)
curve = generate_bezier_trajectory(start, end, direct, num=200)
curve2 = generate_cone_trajectory(start, end, direct, num=200)
rpy_seq = generate_rpy_trajectory([0,0,0],[0.3,-0.1,1.57], len(curve))
```

## 🧪 轨迹生成示例 (通过config配置直接运行)
```bash
python generate.py
python generate_random.py
```

## � scripts 目录脚本使用说明

| 脚本 | 主要功能 | 核心参数 | 常见用法示例 |
|------|----------|----------|--------------|
| `scripts/convert_aligned_to_rl_format.py` | 将单个或多个 `episode*/aligned_joints.h5` 合并转换为 RL / imitation 学习标准 episodic HDF5（含 obs / next_obs / actions / rewards / dones 及可选图片序列） | `--input / --input-dir` 二选一；`--output` 输出路径；`--env-name` 环境名；`--obs-components` / `--action-components`；`--reward-mode`=`default|progress`；`--dataset-type`=`teleop|fieldgen`；`--image-store`=`jpeg|raw`；`--sparse-final-reward`；`--truncate` | 1) 单文件：`python scripts/convert_aligned_to_rl_format.py --input episode0/aligned_joints.h5 --output data/ds.hdf5 --env-name Pick-v0`  2) 多目录：`python scripts/convert_aligned_to_rl_format.py --input-dir processed.tele --output data/ds_multi.hdf5 --env-name Pick-v0 --shuffle --max-episodes 200`  3) 进度奖励：`... --reward-mode progress --ori-weight 0.3` |
| `scripts/create_tele_datasets.py` | 基于一个含多 `episode*` 子目录的数据根，生成 1/5,2/5,...,5/5 递增（或独立）规模数据子集 (tele_k_5) | `--source` 源根目录；`--output` 输出根；`--parts` (默认5)；`--independent` 独立模式；`--force` 覆盖；`--mode`=`copy|symlink|hardlink`；`--prefix` 集合名前缀 | 嵌套递增：`python scripts/create_tele_datasets.py --source processed.tele --output splits --seed 42`  独立集合：`python scripts/create_tele_datasets.py --source processed.tele --output splits --independent --seed 7`  使用硬链接：`... --mode hardlink` |
| `scripts/h5_tree.py` | 以树形结构快速浏览 HDF5 文件层级（不打印数据内容） | 位置参数：文件路径；可选：`-m/--max-depth` 深度；`-i/--show-attr` 显示属性；`-s/--sort` 排序；`--no-color` 关闭颜色 | 基础：`python scripts/h5_tree.py data/ds.hdf5`  限深与属性：`python scripts/h5_tree.py data/ds.hdf5 -m 2 -i`  排序无色：`python scripts/h5_tree.py data/ds.hdf5 -s --no-color` |

### 1. convert_aligned_to_rl_format 详细说明
典型管线：FieldGen 生成的 `episode*/aligned_joints.h5` -> 统一合并 -> 训练框架一次性加载。
```bash
python scripts/convert_aligned_to_rl_format.py \
   --input-dir /path/to/episodes \
   --output data/dataset.hdf5 \
   --env-name CustomPick-v0 \
   --reward-mode progress --ori-weight 0.3 \
   --image-store jpeg --compression gzip --shuffle --seed 42
```
进度奖励模式 `progress` 会依据位置与可选姿态接近度生成密集奖励；若缺必要字段自动回退 default。

可切换 `--dataset-type fieldgen` 直接处理只含 `sample_points.h5` 的目录（会推断缺失 action 并构造）。

### 2. create_tele_datasets 详细说明
用于数据规模曲线实验 (data scaling)。默认嵌套：`tele_3_5` 含 `tele_1_5`,`tele_2_5` 所有 episode 的超集。独立模式适合方差评估。
```bash
python scripts/create_tele_datasets.py \
   --source processed.tele \
   --output splits \
   --parts 5 --seed 42 --force
```
快速仅生成 3 份并独立：`--parts 3 --independent`。

### 3. h5_tree 详细说明
便捷调试数据结构差异：
```bash
python scripts/h5_tree.py data/dataset.hdf5 -m 2 -i -s
```
可配合 `grep` 过滤：`python scripts/h5_tree.py data/dataset.hdf5 | grep demo_0`。

### 命令参数速查
* 输出压缩：`--compression gzip`（体积小，速度较慢） / `lzf`（平衡） / 省略（无压缩）
* 图片存储：`jpeg`（最小）/ `raw`（直接 RGB 数组，读取快）
* 进度奖励权重：`--ori-weight` 控制姿态部分影响 (0~1)
* Episode 采样：`--shuffle --max-episodes N` 控制规模
* 观测/动作组件：默认 `[eef/position, effector/position]`，可扩展传入更多子路径

## �📊 统计输出解读
运行后会打印：
* 全局：处理时间 / 曲线类型 / 成功 episodes / 图像数 / 缺失图像路径
* 各任务：计划处理数量 / 平均曲线长度 (±std) / 截断数 / 扩展数
* 随机版额外：reward 分布（均值 / 百分位 / 直方图）


## 🔄 与训练流水线的衔接
生成的 HDF5 可以作为 imitation / offline RL 数据：
* `action/eef/position` 与 `state/eef/position` 此处相同（可后续叠加噪声）
* `effector` 夹爪：最后一点强制闭合 (0,90) 逻辑可按需求更改
* 可根据 reward/value 增强过滤策略（阈值、top-k 等）

## 📁 目录结构（节选）
```
fieldgen-grasp/
   config/config.yaml
   generate.py
   generate_random.py
   utils/
      bezier_util.py
      cone_util.py
      rpy_util.py
      visualize_points.py
   tests/               # 单元测试(占位)
   requirements.txt
```

## ✅ 测试
当前测试目录为占位，可根据需要添加：
```bash
python -m unittest discover tests
```
（建议：为核心几何函数添加曲线长度单调 / 边界输入 / 退化情况测试）

## ❓ 常见问题 (FAQ)
Q: 曲线过短被大量补齐怎么办？
A: 调大 `beta` (减小采样点数) 或减少 `chunk_size`；也可在补齐时插值而非重复最后一点（可扩展）。

Q: 想让起点附近更密集？
A: 修改 `bezier_util.generate_bezier_trajectory` 中 power < 1，或在 cone 里调整 power 分布逻辑。

Q: reward 太集中？
A: 增加 `endpoint_random_radius`、加入距离项或曲率项；也可改为序列 reward。

Q: 如何支持第三种曲线？
A: 新增 `utils/new_curve_xxx.py` 并在 `generate_curve` 中分支注册。

## 🗺 后续改进建议
* 英文 README / 多语言切换
* 添加 CI + 单元测试覆盖
* 引入 hydra / pydantic 做配置验证
* 更灵活的 gripper profile（非二值闭合）
* 更丰富 reward（轨迹平滑度 / 与障碍距离）

## 🤝 贡献
欢迎提 Issue / PR：
* 新轨迹算法
* 视觉特征对齐 / 图像增强
* 数据格式转换脚本

## 📄 许可证
MIT License

---
如果这个项目对你有帮助，欢迎点一个 ⭐ Star 支持！

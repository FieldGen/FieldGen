

# FieldGen

从稀疏采样的少量末端执行器候选点与姿态出发，自动生成平滑、长度统一、可控分布的接近 / 抓取轨迹集，支持终点随机扰动与简易奖励估计，用于模仿学习与离线强化学习数据扩充。

## 核心价值
在真实或远程采集中，通常只能得到零散关键位姿点（例如人工挑选的兴趣位置）。本工具通过几何构造（Bezier 二次曲线、圆锥内半摆线）与非均匀弧长重采样，将这些稀疏点补全为连续、终端区域高密度的轨迹，并同时生成：
- 姿态（RPY）缓出插值
- 统一长度时间序列（截断 / 补齐）
- 夹爪开合（简单占位，可扩展）
- 原始关联图像拷贝（相机帧）
- 奖励（随机终点模式下的接近度）

## 功能特性
- 轨迹类型：Bezier / Cone（结构化接近路径）
- 终点导向：通过局部 Z 轴方向反向辅助构造曲线
- 非均匀采样：靠近终点区域更密集（学习收敛更稳）
- 长度对齐：按 `chunk_size` 统一，短则补齐，长则截断
- 奖励模式：`--reward` 下多半径扰动采样终点 + 距离归一奖励
- 多任务批处理：读取多个任务目录，分别统计
- 输出格式：标准化 HDF5 分组（action/state/eef/effector/ reward）
- 可视化：点分布三维散点与多视图密度（HTML），轨迹 + 姿态箭头（PNG）
- 配置集中：单一 YAML 文件调参（曲线类型 / 半径 / 路径等）
- 算法简洁：易于扩展第三种轨迹生成方法

## 目录结构（主要）
```
fieldgen-grasp/
   generate.py
   requirements.txt
   config/
      config.yaml
   utils/
      bezier_util.py
      cone_util.py
      rpy_util.py
      visualize_points.py
   scripts/   # 数据格式转换、数据集拆分、HDF5 树浏览等工具
   tests/     # 演示型测试
```

## 安装
建议 Python ≥ 3.10
```bash
git clone <repo-url>
cd fieldgen-grasp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
可选安装交互可视化：
```bash
pip install plotly
```
依赖（`requirements.txt`）：`numpy, scipy, pyyaml, h5py, Pillow, tqdm, matplotlib`（Plotly 可选）

## 配置文件说明 (`config/config.yaml`)
`generate` 部分：
| 字段 | 说明 |
|------|------|
| curve_type | `bezier` 或 `cone` |
| output_path | 生成的 episode 输出根目录 |
| chunk_size | 每条轨迹统一长度 T |
| beta | 二次采样比例控制（用于估计点数） |
| endpoint_random_radius | 随机终点最大扰动半径（reward 模式） |
| multiplier | reward 模式下每原始点扩增倍数 |
| reward_target_min / max | 奖励值裁剪区间（距终点归一后） |

`tasks` 部分：
- 键：任务名称（如 `pick0`）
- `path`: 该任务的源目录（须包含 `sample_points.h5` 与 `camera/*/*.jpg` 可选）
- `max_trajectories`: 限制处理的最大轨迹数（null 表示全部）

示例（截取）：
```yaml
generate:
   curve_type: cone
   output_path: /path/to/output
   chunk_size: 35
   beta: 0.0025
   endpoint_random_radius: 0.1
   multiplier: 10
   reward_target_min: 0.0
   reward_target_max: 1.0

tasks:
   pick0:
      path: /path/to/episode0
      max_trajectories: null
   pick1:
      path: /path/to/episode1
      max_trajectories: null
```

## 输入数据要求
每个任务目录需存在：
```
<task_path>/
   sample_points.h5
   camera/
       0/*.jpg
       1/*.jpg   # 按 eef 索引组织子目录
```
`sample_points.h5` 必备数据集：
- `state/eef/position` : shape `(N, >=12)`，脚本使用列 `[6:9]` 起始 xyz，`[9:12]` 起始 rpy
- `endpoint` : shape `(>=12,)`，使用 `[6:9]` 最终 xyz，`[9:12]` 最终 rpy

## 输出格式
生成的每个 `episodeK/aligned_joints.h5`：
```
timestamps : (T,)
action/eef/position : (T, 12)
state/eef/position  : (T, 12)
action/effector/position : (T, 2)   # 归一化（除以90）
state/effector/position  : (T, 2)
reward/value : 标量或 (T,)        # 仅在 --reward 模式
```
轨迹拼接顺序：`[left_xyz(3), left_rpy(3), right_xyz(3), right_rpy(3)] = 12`。
图像：所有相关 jpg 拷贝到 `episodeK/camera/0/`。

## 使用方法
普通模式（确定终点）：
```bash
python generate.py --config config/config.yaml
```
奖励模式（随机终点 + 奖励写入）：
```bash
python generate.py --config config/config.yaml --reward
```
注意：当前脚本仅支持上述两个参数，早期版本的 `--override-multiplier` / `--override-radius` 已移除。

## 奖励机制说明
Reward 模式下：
1. 读取原始终点位置 `original_endpoint_pos`
2. 在半径 `[0, endpoint_random_radius * (1 - reward_target_min)]` 范围分层采样多个扰动距离
3. 构造方向向量随机扰动终点位置
4. 以扰动距离做归一：`reward = 1 - dist / R_radius`，裁剪到 `[reward_target_min, reward_target_max]`
5. 每个扰动终点生成一条轨迹及对应 `reward/value`

可扩展：加入姿态偏差、轨迹平滑度、障碍距离等复合项。

## 轨迹算法细节
### Bezier 二次曲线
- 控制点：`start`, `end`, 以及对 `start` 投影到 `(end + direct)` 方向上的点
- 弧长估计 + power 重采样：`power > 1` 终段密集；`<1` 起段密集
- 退化（direct≈0）时使用中点作为控制点

### Cone 圆锥内半摆线
- 起点在 cone 内：直接半摆线到顶点
- 起点在 cone 外：轴向直线进入 + half-cycloid 拼接
- 采样点数按直线段 / 摆线估计长度比例再分配

### RPY 姿态插值
- 四元数同半球处理避免绕远
- 二次缓出（ease-out）控制角速度单调下降
- 增量使用轴角累计生成序列

## 可视化
```bash
python utils/visualize_points.py
```
输出：
- `html/point_distribution_3d_<task>.html`
- `html/point_distribution_2d_density_<task>.html`
（浏览器打开）

## 测试（现状）
`tests/` 下为演示型脚本（可视化结果），尚未有严格断言。建议补充：
- 曲线长度与采样稳定性
- 退化输入（start=end / direct≈0）
- RPY 单调角度序列
- reward 分布边界

## 性能与扩展建议
- 大规模任务：图像加载可做惰性或选择性拷贝
- 并行：按任务或按 eef 索引多进程/线程
- 配置验证：可引入 Pydantic / Hydra
- 轨迹扩展：螺旋、最短时间 S 曲线、避障采样
- Reward 扩展：轨迹平滑度、姿态接近度、环境交互指标

## FAQ
| 问题 | 建议 |
|------|------|
| 轨迹过短大量补齐 | 调整 `beta` 或减小 `chunk_size` |
| 终点附近不够密集 | 增大 Bezier/Cone 内部 power 值 |
| 奖励差异太小 | 增大 `endpoint_random_radius` 或降低 `reward_target_min` |
| 添加新轨迹类型 | 新增 `utils/xxx_curve.py` 并修改 `generate_curve` |
| 图像过多影响速度 | 限制每 episode 拷贝数量或移除图像写入 |

## 更新日志
- 2025-10-22: 合并 reward 逻辑入统一脚本；英文注释标准化；简化依赖。

## 许可证
（待补充：在仓库根目录添加 LICENSE 文件，推荐 MIT 或 Apache-2.0）

## 贡献
欢迎提交 Issue / PR：
- 新轨迹算法与采样策略
- 数据转换与格式增强
- 真实单元测试与 CI 集成

## 引用（可选）
后续若发表论文，可在此添加 BibTeX 引用占位。

---
如果该项目对你有帮助，欢迎 Star 支持与分享。
---

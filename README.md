

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

## 测试



运行全部测试用例：
```bash
python -m unittest discover tests
```

## 贡献

欢迎提交 issue 或 pull request。

## 许可证



本项目采用 MIT License。

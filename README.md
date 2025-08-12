

# FieldGen-Grasp

基于 Python 的贝塞尔曲线计算与绘图工具。

## 目录
- [简介](#简介)
- [安装](#安装)
- [使用方法](#使用方法)
- [测试](#测试)
- [贡献](#贡献)
- [许可证](#许可证)

## 简介

本项目包含用于贝塞尔曲线相关计算与绘图的实用工具函数，适合科研、工程和教学用途。

## 安装


建议使用 Python 3.7 及以上版本。

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

## 使用方法


示例：
```python
from utils.bezier_util import bezier_curve
# ...使用 bezier_curve 等函数
```

## 测试


运行测试用例：
```bash
python -m unittest tests/test_bezier_util.py
```

## 贡献

欢迎提交 issue 或 pull request。

## 许可证


本项目采用 MIT License。

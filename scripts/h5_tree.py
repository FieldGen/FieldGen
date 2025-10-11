#!/usr/bin/env python3
"""
打印 HDF5 文件内部层级结构 (仿 tree)。

用法:
  python scripts/h5_tree.py path/to/file.h5

可选参数:
  -m, --max-depth N     限制最大递归深度 (默认: 不限制)
  -i, --show-attr       显示对象(组/数据集)的属性名
  -s, --sort            先按组后按数据集排序 (默认: 读取顺序)
  --no-color            关闭颜色输出

输出说明:
  [G] 代表 Group
  [D] 代表 Dataset (不打印数据内容)

示例:
  python scripts/h5_tree.py data/sample.h5 -m 3 -i -s
"""
from __future__ import annotations
import argparse
import os
import sys
import h5py
from typing import Optional, List

# 简单的 ANSI 颜色支持
class Color:
    BLUE = '\033[34m'
    GREEN = '\033[32m'
    CYAN = '\033[36m'
    RESET = '\033[0m'
    DIM = '\033[2m'

    @staticmethod
    def wrap(text: str, code: str, enable: bool) -> str:
        return f"{code}{text}{Color.RESET}" if enable else text

def format_shape(ds: h5py.Dataset) -> str:
    try:
        shape = ds.shape
        dtype = ds.dtype
        return f"shape={shape}, dtype={dtype}"
    except Exception:
        return "(unavailable)"

def list_children(obj: h5py.Group, sort: bool) -> List[str]:
    names = list(obj.keys())
    if not sort:
        return names
    # 先 group 后 dataset，再按名称
    groups = [n for n in names if isinstance(obj.get(n, getclass=True), h5py.Group)]
    datasets = [n for n in names if isinstance(obj.get(n, getclass=True), h5py.Dataset)]
    others = [n for n in names if n not in groups and n not in datasets]
    return sorted(groups) + sorted(datasets) + sorted(others)

def print_attrs(h5obj, prefix: str, color: bool):
    if not hasattr(h5obj, 'attrs') or len(h5obj.attrs) == 0:
        return
    for k in h5obj.attrs.keys():
        print(f"{prefix}{Color.wrap('@'+k, Color.DIM, color)}")

def walk(name: str, obj, prefix: str, is_last: bool, args, level: int, color: bool):
    branch = '└── ' if is_last else '├── '
    next_prefix = prefix + ('    ' if is_last else '│   ')

    if isinstance(obj, h5py.Group):
        tag = Color.wrap('[G]', Color.BLUE, color)
        print(f"{prefix}{branch}{tag} {name if name else '/'}")
        if args.show_attr:
            print_attrs(obj, next_prefix, color)
        if args.max_depth is not None and level >= args.max_depth:
            return
        children = list_children(obj, args.sort)
        for i, child in enumerate(children):
            child_obj = obj.get(child)
            walk(child, child_obj, next_prefix, i == len(children) - 1, args, level + 1, color)
    elif isinstance(obj, h5py.Dataset):
        tag = Color.wrap('[D]', Color.GREEN, color)
        meta = format_shape(obj)
        print(f"{prefix}{branch}{tag} {name} ({meta})")
        if args.show_attr:
            print_attrs(obj, next_prefix, color)
    else:
        tag = Color.wrap('[?]', Color.CYAN, color)
        print(f"{prefix}{branch}{tag} {name}")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description='以 tree 形式打印 HDF5 结构 (不输出数据内容)')
    parser.add_argument('h5file', help='HDF5 文件路径 (.h5/.hdf5)')
    parser.add_argument('-m', '--max-depth', type=int, default=None, help='最大递归深度 (根为 0)')
    parser.add_argument('-i', '--show-attr', action='store_true', help='显示对象属性名')
    parser.add_argument('-s', '--sort', action='store_true', help='按组/数据集分类并排序')
    parser.add_argument('--no-color', action='store_true', help='关闭彩色输出')
    args = parser.parse_args(argv)

    if not os.path.isfile(args.h5file):
        print(f"错误: 找不到文件 {args.h5file}", file=sys.stderr)
        return 1

    color = not args.no_color and sys.stdout.isatty()

    try:
        with h5py.File(args.h5file, 'r') as f:
            # 根节点
            root_name = ''
            walk(root_name, f, prefix='', is_last=True, args=args, level=0, color=color)
    except OSError as e:
        print(f"打开文件失败: {e}", file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())

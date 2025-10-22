#!/usr/bin/env python3
"""
Print an HDF5 file's internal hierarchy (tree-like view).

Usage:
    python scripts/h5_tree.py path/to/file.h5

Optional arguments:
    -m, --max-depth N     Limit max recursion depth (default: unlimited)
    -i, --show-attr       Show attribute names for groups/datasets
    -s, --sort            Sort: groups first, then datasets (default: raw order)
    --no-color            Disable ANSI color output

Legend:
    [G] Group
    [D] Dataset (data contents are not printed)

Example:
    python scripts/h5_tree.py data/sample.h5 -m 3 -i -s
"""
from __future__ import annotations
import argparse
import os
import sys
import h5py
from typing import Optional, List

# Simple ANSI color support
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
    # Sort: groups first, then datasets, then others, each name ascending
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
    parser = argparse.ArgumentParser(description='Print HDF5 structure in a tree format (no data contents)')
    parser.add_argument('h5file', help='Path to HDF5 file (.h5/.hdf5)')
    parser.add_argument('-m', '--max-depth', type=int, default=None, help='Maximum recursion depth (root=0)')
    parser.add_argument('-i', '--show-attr', action='store_true', help='Show attribute names')
    parser.add_argument('-s', '--sort', action='store_true', help='Sort groups/datasets')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    args = parser.parse_args(argv)

    if not os.path.isfile(args.h5file):
        print(f"Error: file not found {args.h5file}", file=sys.stderr)
        return 1

    color = not args.no_color and sys.stdout.isatty()

    try:
        with h5py.File(args.h5file, 'r') as f:
            # Root node
            root_name = ''
            walk(root_name, f, prefix='', is_last=True, args=args, level=0, color=color)
    except OSError as e:
        print(f"Failed to open file: {e}", file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())

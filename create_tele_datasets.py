#!/usr/bin/env python3
"""
根据已有 episode 目录，生成按比例递增的 5 个数据集：
tele_1_5, tele_2_5, tele_3_5, tele_4_5, tele_5_5

特点 / 规则：
1. 源目录下假定存在形如 episode{index} 的子目录（忽略非匹配与 'separated'）。
2. 先对所有 episode 做一次随机洗牌（--seed 控制可复现）。
3. 生成递增嵌套子集（默认）：第 k 份包含 ceil(N * k / 5) 个 episode，是第 k-1 份的超集。
4. 拷贝到输出根目录下对应数据集文件夹内，并重新按 0..n-1 连续重命名为 episode0, episode1, ...
5. 若目录已存在：默认报错，可用 --force 覆盖（删除后重建）。
6. 可选模式：--independent 使 5 份彼此独立随机抽样（各自单独洗牌 + 取前 ceil(N * k / 5)）。
7. 支持选择拷贝方式：copy(默认) / symlink / hardlink。

用法示例：
python create_tele_datasets.py \
  --source /mnt/yekehe/fieldgen_mainexp1/pick/processed.pick.tele \
  --output /mnt/yekehe/fieldgen_mainexp1/pick \
  --seed 42

可选独立：加 --independent
可选覆盖：加 --force

作者：自动生成脚本
"""
from __future__ import annotations
import argparse
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path

EPISODE_DIR_PATTERN = re.compile(r"^episode(\d+)$")

def collect_episodes(source: Path) -> list[Path]:
    if not source.is_dir():
        raise ValueError(f"源目录不存在: {source}")
    eps = []
    for child in sorted(source.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name == 'separated':
            continue
        if EPISODE_DIR_PATTERN.match(name):
            eps.append(child)
    if not eps:
        raise ValueError(f"未找到任何 episode* 目录于: {source}")
    return eps


def ensure_clean_dir(path: Path, force: bool):
    if path.exists():
        if not force:
            raise FileExistsError(f"目标目录已存在: {path} (使用 --force 覆盖)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_episode(src: Path, dst: Path, mode: str):
    if mode == 'copy':
        shutil.copytree(src, dst)
    elif mode == 'symlink':
        # 软链接整个目录（兼容性：某些工具可能不期望目录是符号链接）
        os.symlink(src, dst, target_is_directory=True)
    elif mode == 'hardlink':
        # 递归硬链接：需逐文件复制结构
        if dst.exists():
            raise FileExistsError(dst)
        dst.mkdir(parents=True)
        for root, dirs, files in os.walk(src):
            rel = Path(root).relative_to(src)
            cur_dst = dst / rel
            for d in dirs:
                (cur_dst / d).mkdir(exist_ok=True)
            for f in files:
                src_file = Path(root) / f
                dst_file = cur_dst / f
                os.link(src_file, dst_file)
    else:
        raise ValueError(f"未知复制模式: {mode}")


def build_nested_indices(total: int, parts: int) -> list[int]:
    # 返回每个部分的累计目标数量
    return [math.ceil(total * (k + 1) / parts) for k in range(parts)]


def generate_datasets(source: Path, output: Path, parts: int, seed: int | None,
                      force: bool, independent: bool, mode: str, prefix: str):
    episodes = collect_episodes(source)
    total = len(episodes)
    print(f"发现 {total} 个 episodes 于 {source}")

    part_sizes_cum = build_nested_indices(total, parts)

    if independent:
        print("使用独立 (non-nested) 抽样模式。")
    else:
        print("使用嵌套 (nested incremental) 抽样模式。")

    rng = random.Random(seed)

    if not independent:
        shuffled = episodes[:]
        rng.shuffle(shuffled)

    for idx, cum_n in enumerate(part_sizes_cum, start=1):
        dataset_name = f"{prefix}_{idx}_{parts}"
        dataset_dir = output / dataset_name
        ensure_clean_dir(dataset_dir, force)

        if independent:
            local = episodes[:]
            rng.shuffle(local)
            selected = local[:cum_n]
        else:
            selected = shuffled[:cum_n]

        print(f"构建 {dataset_name}: 选取 {len(selected)} 条 episode")

        # 重新编号复制
        for new_i, ep_path in enumerate(selected):
            new_dir = dataset_dir / f"episode{new_i}"
            copy_episode(ep_path, new_dir, mode)

    print("完成。")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="按比例生成递增或独立 episode 数据子集")
    p.add_argument('--source', type=Path, required=True, help='源 episodes 根目录 (包含 episode0, episode1, ...)')
    p.add_argument('--output', type=Path, required=True, help='输出根目录 (将在其中创建 tele_*_* 子目录)')
    p.add_argument('--parts', type=int, default=5, help='分成多少等份 (默认 5)')
    p.add_argument('--seed', type=int, default=42, help='随机种子 (默认 42)')
    p.add_argument('--force', action='store_true', help='若目标子目录存在则删除重建')
    p.add_argument('--independent', action='store_true', help='生成互相独立而非嵌套的子集')
    p.add_argument('--mode', choices=['copy', 'symlink', 'hardlink'], default='copy', help='拷贝模式')
    p.add_argument('--prefix', default='tele', help='数据集名前缀 (默认 tele)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    generate_datasets(
        source=args.source,
        output=args.output,
        parts=args.parts,
        seed=args.seed,
        force=args.force,
        independent=args.independent,
        mode=args.mode,
        prefix=args.prefix,
    )

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Generate proportional incremental subsets of existing episode directories:
tele_1_5, tele_2_5, tele_3_5, tele_4_5, tele_5_5

Rules / Behavior:
1. Source root is expected to contain subdirectories named episode{index} (others and 'separated' are ignored).
2. All episodes are randomly shuffled first (controlled by --seed for reproducibility).
3. Default nested mode: subset k contains ceil(N * k / parts) episodes and is a superset of subset k-1.
4. Copies (or links) selected episodes into output root under each dataset folder, renaming sequentially to episode0..episode{n-1}.
5. If a target dataset directory exists: raise by default; use --force to remove and rebuild.
6. Optional independent mode (--independent): each of the k subsets samples independently (own shuffle + take first ceil(N * k / parts)).
7. Copy strategy selectable: copy (default) / symlink / hardlink.

Example:
python create_tele_datasets.py \
    --source /mnt/yekehe/fieldgen_mainexp1/pick/processed.pick.tele \
    --output /mnt/yekehe/fieldgen_mainexp1/pick \
    --seed 42

Optional independent mode: add --independent
Optional overwrite: add --force
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
        raise ValueError(f"Source directory not found: {source}")
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
        raise ValueError(f"No episode* directories found in: {source}")
    return eps


def ensure_clean_dir(path: Path, force: bool):
    if path.exists():
        if not force:
            raise FileExistsError(f"Target directory exists: {path} (use --force to overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_episode(src: Path, dst: Path, mode: str):
    if mode == 'copy':
        shutil.copytree(src, dst)
    elif mode == 'symlink':
        # Symlink whole directory (note: some tools may not expect symlinked dirs)
        os.symlink(src, dst, target_is_directory=True)
    elif mode == 'hardlink':
        # Recursive hardlink: reproduce directory tree, link each file
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
        raise ValueError(f"Unknown copy mode: {mode}")


def build_nested_indices(total: int, parts: int) -> list[int]:
    # Return cumulative target counts for each subset part
    return [math.ceil(total * (k + 1) / parts) for k in range(parts)]


def generate_datasets(source: Path, output: Path, parts: int, seed: int | None,
                      force: bool, independent: bool, mode: str, prefix: str):
    episodes = collect_episodes(source)
    total = len(episodes)
    print(f"Found {total} episodes in {source}")

    part_sizes_cum = build_nested_indices(total, parts)

    if independent:
        print("Independent (non-nested) sampling mode.")
    else:
        print("Nested incremental sampling mode.")

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

        # Re-number and copy episodes into new dataset folder
        for new_i, ep_path in enumerate(selected):
            new_dir = dataset_dir / f"episode{new_i}"
            copy_episode(ep_path, new_dir, mode)

    print("Done.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate proportional incremental or independent episode subsets")
    p.add_argument('--source', type=Path, required=True, help='Source episodes root (contains episode0, episode1, ...)')
    p.add_argument('--output', type=Path, required=True, help='Output root (tele_*_* directories will be created here)')
    p.add_argument('--parts', type=int, default=5, help='Number of proportional parts (default 5)')
    p.add_argument('--seed', type=int, default=42, help='Random seed (default 42)')
    p.add_argument('--force', action='store_true', help='Overwrite existing dataset directories if present')
    p.add_argument('--independent', action='store_true', help='Generate independent (non-nested) subsets')
    p.add_argument('--mode', choices=['copy', 'symlink', 'hardlink'], default='copy', help='Copy/link mode')
    p.add_argument('--prefix', default='tele', help='Dataset name prefix (default tele)')
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

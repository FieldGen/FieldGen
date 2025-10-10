#!/usr/bin/env python3
"""
将单一或多个 episode 目录中的 HDF5 文件 (aligned_joints.h5) 转换为 RL / imitation 学习常用的 episodic 结构：

单文件输入格式 (示例)：
  aligned_joints.h5
  ├── timestamps                 # shape (T,) int/float 递增
  ├── action/
  │   ├── eef/position           # shape (T, 12)
  │   └── effector/position      # shape (T, 2)  (夹爪动作，已缩放：gripper/90)
  └── state/
      ├── eef/position           # shape (T, 12)  (与 action 中一致)
      └── effector/position      # shape (T, 2)   (原始夹爪值，末步可能为 [0,90])

目录输入格式 (示例)：
    dataset_root/
        episode0/aligned_joints.h5
        episode1/aligned_joints.h5
        ...

输出格式：
  output_root/
    data.hdf5  (或者自定义文件名)
      attrs:
    total: <int>                # 总 demo 数（单文件=1；目录=episode 数）
        env_args: <json str>        # 至少包含 env 名与参数，可通过命令行传入 --env-name / --env-args-json
      demo_0/
        attrs:
          num_samples: T
          (可选) model_file: (若通过 --model-file 传入)
        obs/flat         (T, ObsDim)
        next_obs/flat    (T, ObsDim)  # 最后一帧 = obs 最后一帧 (终止补拷贝)
        actions          (T, A)
        rewards          (T,) float32 （当前默认全 0，可通过 --sparse-final-reward 自定义）
        dones            (T,) uint8  (最后一帧=1，其余 0)

Obs / Action 设计：
  这里给出一个简单基线：
    obs = concat( state/eef/position (12), state/effector/position (2) ) => 14 维
    action = concat( action/eef/position (12), action/effector/position (2) ) => 14 维
  你可以根据需要自定义：支持通过 --obs-components 与 --action-components 控制使用哪些子路径。

使用示例：
    # 单文件
    python scripts/convert_aligned_to_rl_format.py \
        --input aligned_joints.h5 \
        --output data/dataset.hdf5 \
        --env-name CustomPickPlace-v0 \
        --env-args-json '{"max_steps":200}'

    # 多 episode 目录（目录内含 episode*/aligned_joints.h5）
    python scripts/convert_aligned_to_rl_format.py \
        --input-dir /path/to/processed.mainexp2.tele \
        --output data/dataset_multi.hdf5 \
        --env-name CustomPickPlace-v0 \
        --shuffle --seed 42 --max-episodes 100

可选：
  --model-file path/to/model.xml
  --sparse-final-reward 1.0   # 给最后一步一个稀疏奖励
  --reward-per-step 0.0       # 每步基础奖励（与 sparse 可叠加）
  --copy-gripper-from-state   # 若想以 state effector 作为 obs/action 的 effector 部分（默认 action effector 用于 action, state effector 用于 obs）
    --compression gzip          # 使用 gzip 压缩 (可选 lzf, gzip, none)
    --image-store raw|jpeg      # 图片存储：raw=解码后RGB (占空间大)，jpeg=原始JPEG字节(最省空间)
    --reward-mode default|progress  # progress 使用曲线进度奖励 (1 - 剩余长度/总长度)

Edge Cases:
  - 自动检查各通道时间长度一致。
  - 自动将数据转换为 float32 (动作/观测)，时间戳保持原 dtype。
  - 若提供 --truncate T0 可截断前 T0 步。

Author: Auto-generated
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Iterable, Tuple
from PIL import Image
import re

# ------------------------------------------------------------------
# 确保可以从项目根目录导入 utils (直接运行 scripts/ 下脚本时 sys.path 只包含 scripts)
# ------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.offline_util import compute_progress_rewards  # noqa: E402
import h5py  # noqa: E402
from tqdm import tqdm  # noqa: E402
import numpy as np  # noqa: E402

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------

def read_required(ds: h5py.Group, path: str):
    if path not in ds:
        raise KeyError(f"缺少数据集: {path}")
    return ds[path][...]


def build_array(root: h5py.File, base: str, components: List[str]) -> np.ndarray:
    parts = []
    for comp in components:
        p = f"{base}/{comp}"
        if p not in root:
            raise KeyError(f"未找到组件 {p}")
        arr = root[p][...]
        parts.append(arr)
    # 验证时间长度一致
    T_set = {a.shape[0] for a in parts}
    if len(T_set) != 1:
        raise ValueError(f"组件时间长度不一致: {[a.shape for a in parts]}")
    return np.concatenate(parts, axis=-1).astype(np.float32)


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def create_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        return parent[name]
    return parent.create_group(name)


def write_dataset(g: h5py.Group, name: str, data: np.ndarray, compression: str | None):
    if name in g:
        del g[name]
    kwargs = {}
    if compression and compression.lower() != 'none':
        kwargs['compression'] = compression
    g.create_dataset(name, data=data, **kwargs)

# -----------------------------------------------------
# Main conversion
# -----------------------------------------------------

def process_single_file(h5_path: Path, obs_components: List[str], action_components: List[str],
                        copy_gripper_from_state: bool, truncate: int | None,
                        dataset_type: str = 'teleop') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取单个 aligned_joints.h5 返回 (obs, next_obs, actions, rewards, dones) (rewards 与 dones 先留空位)

    注意：此函数只处理数值序列，不处理图片；图片在更外层根据 episode 目录附加。
    """
    with h5py.File(h5_path, 'r') as f_in:
        if dataset_type == 'teleop':
            # 原始逻辑
            obs_arr = build_array(f_in, 'state', obs_components)
            action_arr = build_array(f_in, 'action', action_components)
            if copy_gripper_from_state and 'effector/position' in action_components and 'effector/position' in obs_components:
                # 重新打开方式略低效，但保持最小侵入
                dims = [h5py.File(h5_path, 'r')[f"action/{comp}"].shape[-1] for comp in action_components]
                dims_obs = [h5py.File(h5_path, 'r')[f"state/{comp}"].shape[-1] for comp in obs_components]
                idx_eff = action_components.index('effector/position')
                idx_eff_obs = obs_components.index('effector/position')
                start = sum(dims[:idx_eff]); end = start + dims[idx_eff]
                start_obs = sum(dims_obs[:idx_eff_obs]); end_obs = start_obs + dims_obs[idx_eff_obs]
                action_arr[:, start:end] = obs_arr[:, start_obs:end_obs]
        elif dataset_type == 'fieldgen':
            # fieldgen: 可能只有 state/eef/position (+ 可选 endpoint / timestamps)，没有 action/* 或 effector/*
            # 策略：
            #   - 对于 state 下缺失的组件，用 0 填充
            #   - action 与 obs 维度一致，若缺失 action 组则用 0
            parts_obs = []
            obs_dims = []
            for comp in obs_components:
                path = f'state/{comp}'
                if path in f_in:
                    arr = f_in[path][...].astype(np.float32)
                else:
                    # 推断长度: 若 comp == 'eef/position' 尝试使用 state/eef/position；若缺失则报错
                    if comp == 'eef/position' and 'state/eef/position' in f_in:
                        arr = f_in['state/eef/position'][...].astype(np.float32)
                    else:
                        # 需要根据已有长度 T 构造
                        if parts_obs:
                            T_local = parts_obs[0].shape[0]
                        else:
                            # 如果第一个组件就缺失且不是 eef/position 无法推断 T
                            raise KeyError(f'fieldgen 模式缺失必要组件，无法推断时间长度: {path}')
                        # 默认长度 (effector/position) 2
                        fill_dim = 2 if comp == 'effector/position' else 1
                        arr = np.zeros((T_local, fill_dim), dtype=np.float32)
                parts_obs.append(arr)
                obs_dims.append(arr.shape[-1])
            # 验证时间长度
            T_set = {a.shape[0] for a in parts_obs}
            if len(T_set) != 1:
                raise ValueError('fieldgen 模式下 obs 组件时间长度不一致')
            obs_arr = np.concatenate(parts_obs, axis=-1)

            # 动作：若 action 组存在则与 teleop 一致，否则 zeros
            if 'action' in f_in:
                parts_action = []
                for comp in action_components:
                    path = f'action/{comp}'
                    if path in f_in:
                        parts_action.append(f_in[path][...].astype(np.float32))
                    elif comp == 'eef/position' and 'action/eef/position' in f_in:
                        parts_action.append(f_in['action/eef/position'][...].astype(np.float32))
                    else:
                        # fabricate zeros matching obs counterpart dimension
                        dim = obs_dims[action_components.index(comp)] if comp in obs_components else 1
                        parts_action.append(np.zeros((obs_arr.shape[0], dim), dtype=np.float32))
                action_arr = np.concatenate(parts_action, axis=-1)
            else:
                # 没有 action 组：用“shift 后的 state” 近似动作，保持同长度，不丢最后一帧
                # action[t] = obs[t+1] (最后一帧用自身或 0 填充)
                action_arr = np.empty_like(obs_arr)
                action_arr[:-1] = obs_arr[1:]
                action_arr[-1] = obs_arr[-1]  # 或者置 0: np.zeros_like(obs_arr[-1])
                # 若更希望使用差分(速度)，可改为：
                # diff = np.zeros_like(obs_arr)
                # diff[:-1] = obs_arr[1:] - obs_arr[:-1]
                # action_arr = diff
        else:
            raise ValueError(f'未知 dataset_type: {dataset_type}')

        T = obs_arr.shape[0]
        if truncate is not None:
            T = min(T, truncate)
            obs_arr = obs_arr[:T]; action_arr = action_arr[:T]
        next_obs = np.concatenate([obs_arr[1:], obs_arr[-1:]], axis=0)
        rewards = np.empty((T,), dtype=np.float32)
        dones = np.zeros((T,), dtype=np.uint8); dones[-1] = 1
    return obs_arr, next_obs, action_arr, rewards, dones


def iter_episode_files(root_dir: Path, dataset_type: str = 'teleop') -> Iterable[Path]:
    """迭代 episode H5 文件。

    teleop: 期望 episode*/aligned_joints.h5
    fieldgen: 若存在 episode* 目录同样使用；否则尝试直接寻找 aligned_joints.h5 或 sample_points.h5 作为单一输入。
    """
    file_name = 'sample_points.h5' if dataset_type == 'fieldgen' else 'aligned_joints.h5'
    for child in sorted(root_dir.iterdir()):
        if child.is_dir() and child.name.startswith('episode'):
            h5_path = child / file_name
            if h5_path.exists():
                yield h5_path


def convert(input_path: Path | None, input_dir: Path | None, output_path: Path, env_name: str, env_args_json: str | None,
            model_file: str | None, obs_components: List[str], action_components: List[str],
            reward_per_step: float, sparse_final_reward: float | None,
            copy_gripper_from_state: bool, truncate: int | None, compression: str | None,
            image_store: str,
            max_episodes: int | None, shuffle: bool, seed: int | None,
            reward_mode: str = 'default',
            ori_weight: float = 0.0,
            dataset_type: str = 'teleop'):
    if (input_path is None) == (input_dir is None):
        raise ValueError('必须且只能提供 --input 或 --input-dir 之一')
    ensure_dir(output_path)

    # 收集 episodes
    episode_files: list[Path]
    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(input_dir)
        episode_files = list(iter_episode_files(input_dir, dataset_type=dataset_type))
        if not episode_files:
            raise ValueError(f'目录 {input_dir} 下未找到任何可用 H5 (dataset_type={dataset_type})')
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(episode_files)
        if max_episodes is not None:
            episode_files = episode_files[:max_episodes]
    else:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        episode_files = [input_path]

    env_args = {"env_name": env_name}
    if env_args_json:
        env_args.update(json.loads(env_args_json))
    env_args_str = json.dumps(env_args, ensure_ascii=False)

    with h5py.File(output_path, 'w') as f_out:
        f_out.attrs['total'] = len(episode_files)
        f_out.attrs['env_args'] = env_args_str

        pbar = tqdm(total=len(episode_files), desc='demos', unit='demo')
        for epi_idx, h5_file in enumerate(episode_files):
            obs_arr, next_obs, action_arr, rewards, dones = process_single_file(
                h5_file, obs_components, action_components, copy_gripper_from_state, truncate, dataset_type=dataset_type
            )
            T = obs_arr.shape[0]
            # 奖励计算
            if reward_mode == 'progress':
                # progress 奖励：基于 state/eef/position 第 6:9 列与 endpoint (若存在)
                endpoint = None
                try:
                    with h5py.File(h5_file, 'r') as src:
                        if 'state/eef/position' in src:
                            state_pos_full = src['state/eef/position'][...]
                        else:
                            raise KeyError('缺少 state/eef/position 用于 progress 奖励')
                        if 'endpoint' in src:
                            endpoint_arr = src['endpoint'][...]
                            # teleop 常见 endpoint 可能为 (3,)；fieldgen 为 (12,)
                            if endpoint_arr.shape == (12,):
                                endpoint = endpoint_arr.astype(np.float32)[6:12]
                            elif endpoint_arr.shape == (6,):
                                endpoint = endpoint_arr.astype(np.float32)
                except Exception as e:
                    print(f"警告: progress 奖励读取失败，回退 default: {e}")
                    reward_mode_effective = 'default'
                else:
                    # state_pos_full shape (T,12) -> 取 6:9
                    pts = state_pos_full[:T, 6:9]
                    # 提取 RPY 序列（假设 9:12 为 rpy）
                    rpy_seq = None
                    rpy_end = None
                    if state_pos_full.shape[1] >= 12:
                        rpy_seq = state_pos_full[:T, 9:12]
                    if endpoint is not None and endpoint.shape[0] >= 6:
                        rpy_end = endpoint[3:6]
                    progress_r = compute_progress_rewards(pts, endpoint=endpoint, rpy_seq=rpy_seq, rpy_end=rpy_end, ori_weight=ori_weight)[:T]
                    rewards[:] = progress_r.astype(np.float32)
                    reward_mode_effective = 'progress'
                if reward_mode_effective != 'progress':
                    rewards.fill(reward_per_step)
                    if sparse_final_reward is not None:
                        rewards[-1] += np.float32(sparse_final_reward)
            else:
                rewards.fill(reward_per_step)
                if sparse_final_reward is not None:
                    rewards[-1] += np.float32(sparse_final_reward)
            demo = create_group(f_out, f'demo_{epi_idx}')
            demo.attrs['num_samples'] = T
            if model_file:
                demo.attrs['model_file'] = model_file
            g_obs = create_group(demo, 'obs')
            g_next = create_group(demo, 'next_obs')
            write_dataset(g_obs, 'flat', obs_arr, compression)
            write_dataset(g_next, 'flat', next_obs, compression)
            write_dataset(demo, 'actions', action_arr, compression)
            write_dataset(demo, 'rewards', rewards, compression)
            write_dataset(demo, 'dones', dones, compression)

            # 附加相机多帧图片序列：episode_dir/camera/<frame_idx>/<view>.jpg
            # view 目标: hand_left.jpg, hand_right.jpg, head.jpg
            episode_dir = h5_file.parent
            camera_root = episode_dir / 'camera'
            frame_dir_pattern = re.compile(r'^\d+$')
            if camera_root.exists():
                frame_dirs = [d for d in sorted(camera_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name)
                               if d.is_dir() and frame_dir_pattern.match(d.name)]
                views = ['hand_left.jpg', 'hand_right.jpg', 'head.jpg']
                for view in views:
                    if image_store == 'jpeg':
                        # 以原始 JPEG 字节方式存储，可变长度 uint8
                        jpeg_bytes_list = []
                        for fd in frame_dirs:
                            img_path = fd / view
                            if not img_path.exists():
                                continue
                            try:
                                with open(img_path, 'rb') as fh:
                                    data_bytes = fh.read()
                                jpeg_bytes_list.append(np.frombuffer(data_bytes, dtype=np.uint8))
                            except Exception as e:
                                print(f"警告: 读取图片失败 {img_path}: {e}")
                        if not jpeg_bytes_list:
                            continue
                        ds_name = view[:-4] if view.endswith('.jpg') else view
                        if ds_name in g_obs:
                            del g_obs[ds_name]
                        vlen_dtype = h5py.vlen_dtype(np.uint8)
                        ds = g_obs.create_dataset(ds_name, (len(jpeg_bytes_list),), dtype=vlen_dtype)
                        ds[...] = jpeg_bytes_list
                        ds.attrs['format'] = 'jpeg-bytes'
                        # 记录第一帧尺寸 (lazy decode for metadata)
                        try:
                            # 仅解码第一帧获取尺寸
                            from io import BytesIO
                            with Image.open(BytesIO(jpeg_bytes_list[0].tobytes())) as im0:
                                w0, h0 = im0.size
                            ds.attrs['width'] = w0
                            ds.attrs['height'] = h0
                        except Exception:
                            pass
                    else:
                        # raw 模式：解码为 RGB ndarray 堆叠
                        imgs_meta = []  # (path, arr)
                        for fd in frame_dirs:
                            img_path = fd / view
                            if not img_path.exists():
                                continue
                            try:
                                with Image.open(img_path) as im:
                                    arr = np.asarray(im.convert('RGB'))
                                imgs_meta.append((img_path, arr))
                            except Exception as e:
                                print(f"警告: 读取图片失败 {img_path}: {e}")
                        if not imgs_meta:
                            continue
                        heights = {a.shape[0] for _,a in imgs_meta}
                        widths = {a.shape[1] for _,a in imgs_meta}
                        if len(heights) == 1 and len(widths) == 1:
                            stack = np.stack([a for _,a in imgs_meta], axis=0).astype(np.uint8)
                        else:
                            min_h = min(heights); min_w = min(widths)
                            proc = []
                            for _, a in imgs_meta:
                                h,w = a.shape[:2]
                                if h!=min_h or w!=min_w:
                                    y0 = (h - min_h)//2; x0 = (w - min_w)//2
                                    a = a[y0:y0+min_h, x0:x0+min_w]
                                proc.append(a)
                            stack = np.stack(proc, axis=0).astype(np.uint8)
                        ds_name = view[:-4] if view.endswith('.jpg') else view
                        if ds_name in g_obs:
                            del g_obs[ds_name]
                        g_obs.create_dataset(ds_name, data=stack, compression=compression if compression and compression!='none' else None)
            # 更新进度条描述
            pbar.set_postfix_str(f"demo_{epi_idx} T={T}")
            pbar.update(1)
        pbar.close()

    print(f"转换完成：{len(episode_files)} demos -> {output_path}")

# -----------------------------------------------------
# CLI
# -----------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='将单文件或多 episode 目录中的 aligned_joints.h5 转换为 RL episodic 数据集')
    p.add_argument('--input', type=Path, help='输入单个 HDF5 文件路径 (aligned_joints.h5)')
    p.add_argument('--input-dir', type=Path, help='输入目录（包含 episode*/aligned_joints.h5）')
    p.add_argument('--output', type=Path, required=True, help='输出 HDF5 文件路径 (如 data/dataset.hdf5)')
    p.add_argument('--env-name', type=str, required=True, help='环境名称')
    p.add_argument('--env-args-json', type=str, default=None, help='额外环境参数 JSON 字符串')
    p.add_argument('--model-file', type=str, default=None, help='可选：模型文件路径写入 attrs')
    p.add_argument('--obs-components', type=str, nargs='+', default=['eef/position', 'effector/position'],
                   help='观测组件（state 下的子路径列表）')
    p.add_argument('--action-components', type=str, nargs='+', default=['eef/position', 'effector/position'],
                   help='动作组件（action 下的子路径列表）')
    p.add_argument('--reward-per-step', type=float, default=0.0, help='每步基础奖励')
    p.add_argument('--sparse-final-reward', type=float, default=1.0, help='末步稀疏奖励')
    p.add_argument('--copy-gripper-from-state', action='store_true', help='用 state effector 覆盖 action effector')
    p.add_argument('--truncate', type=int, default=None, help='截断时间步（可选）')
    p.add_argument('--compression', type=str, default=None, help='HDF5 压缩：gzip/lzf/none')
    p.add_argument('--max-episodes', type=int, default=None, help='最多转换多少个 episode')
    p.add_argument('--image-store', type=str, default='jpeg', choices=['raw','jpeg'], help='图片存储方式：raw=解码后RGB数组, jpeg=原始JPEG字节 (节省空间)')
    p.add_argument('--shuffle', action='store_true', help='对 episode 顺序随机洗牌')
    p.add_argument('--seed', type=int, default=42, help='随机种子(用于 shuffle)')
    p.add_argument('--reward-mode', type=str, default='default', choices=['default','progress'], help='奖励模式: default=原逻辑 progress=位置/姿态进度奖励')
    p.add_argument('--ori-weight', type=float, default=0.3, help='progress 模式下姿态进度线性权重 (0~1), 0 表示不使用姿态')
    p.add_argument('--dataset-type', type=str, default='teleop', choices=['teleop','fieldgen'], help='数据集类型：teleop=原始示教数据; fieldgen=场景生成(sample_points或生成的episode)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    convert(
        input_path=args.input,
        input_dir=args.input_dir,
        output_path=args.output,
        env_name=args.env_name,
        env_args_json=args.env_args_json,
        model_file=args.model_file,
        obs_components=args.obs_components,
        action_components=args.action_components,
        reward_per_step=args.reward_per_step,
        sparse_final_reward=args.sparse_final_reward,
        copy_gripper_from_state=args.copy_gripper_from_state,
        truncate=args.truncate,
        compression=args.compression,
        max_episodes=args.max_episodes,
        shuffle=args.shuffle,
        seed=args.seed,
        image_store=args.image_store,
        reward_mode=args.reward_mode,
        ori_weight=args.ori_weight,
        dataset_type=args.dataset_type,
    )

if __name__ == '__main__':
    main()

import os
import yaml
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import argparse

from utils.bezier_util import generate_bezier_trajectory
from utils.cone_util import generate_cone_trajectory
from utils.rpy_util import generate_rpy_trajectory
from scipy.spatial.transform import Rotation as R

def curve_length(curve):
    return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))

def get_direct(rpy_end):
    rpy_end = np.asarray(rpy_end, dtype=float)
    if rpy_end.shape != (3,):
        raise ValueError("rpy_end 必须是长度为3的一维数组")
    rot = R.from_euler('xyz', rpy_end)
    forward = rot.apply(np.array([0.0, 0.0, 1.0]))  # 本地Z轴
    vec = -forward
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def generate_curve(curve_type, start, end, rpy_end, beta, maxn = 2000):
    direct = get_direct(rpy_end)
    if curve_type == 'bezier':
        curve0 = generate_bezier_trajectory(start, end, direct, num=maxn)
        curve0_length = curve_length(curve0)
        curve = generate_bezier_trajectory(start, end, direct, num=int(curve0_length/beta))
    elif curve_type == 'cone':
        curve0 = generate_cone_trajectory(start, end, direct, num=maxn)
        curve0_length = curve_length(curve0)
        curve = generate_cone_trajectory(start, end, direct, num=int(curve0_length/beta))
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}")
    
    return curve

def generate_episode(output_dir, episode_id, combined_data, gripper, imgs, reward_value=None):
    episode_path = os.path.join(output_dir, f"episode{episode_id}")
    os.makedirs(episode_path, exist_ok=True)

    # Ensure camera/0 directory exists
    camera_path = os.path.join(episode_path, 'camera', '0')
    os.makedirs(camera_path, exist_ok=True)
    # Save the images using their keys
    for img_key, img in imgs.items():
        img_filename = f"{img_key}.jpg"
        img_path = os.path.join(camera_path, img_filename)
        img.save(img_path)

    h5_path = os.path.join(episode_path, 'aligned_joints.h5')
    with h5py.File(h5_path, 'w') as h5_file:
        h5_file.create_dataset('timestamps', data=np.arange(len(combined_data)))
        g_action = h5_file.create_group('action')
        g_state = h5_file.create_group('state')
        g_action_eef = g_action.create_group('eef')
        g_state_eef = g_state.create_group('eef')
        g_action_gripper = g_action.create_group('effector')
        g_state_gripper = g_state.create_group('effector')
        if reward_value is not None:
            g_reward = h5_file.create_group('reward')
            g_reward.create_dataset('value', data=reward_value)

        g_action_eef.create_dataset('position', data=combined_data)
        g_state_eef.create_dataset('position', data=combined_data)
        g_action_gripper.create_dataset('position', data=gripper/90)
        g_state_gripper.create_dataset('position', data=gripper)

    # visualize_curve_with_rpy(combined_data, episode_path)

    # print(f"Episode {episode_id} generated: {episode_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Unified trajectory generation script (supports optional --reward mode)")
    parser.add_argument("--config", type=str, default=os.path.join('config', 'config.yaml'), help="Path to configuration YAML file")
    parser.add_argument("--reward", action="store_true", help="Enable stochastic endpoint sampling (reward mode)")
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    
    # Initialize simplified global statistics and per-task stats container
    stats = {
        'total_eef_positions': 0,
        'total_successful_episodes': 0,
        'total_images_processed': 0,
        'total_missing_image_paths': 0,
        'curve_type_used': '',
        'chunk_size_used': 0,
    }

    per_task_stats = {}
    
    # Load configuration
    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    gen_cfg = config.get('generate', {})
    curve_type = gen_cfg.get('curve_type', 'bezier')
    chunk_size = gen_cfg.get('chunk_size', 32)
    output_path = gen_cfg.get('output_path', 'output/')
    beta = gen_cfg.get('beta', 0.003)
    endpoint_random_radius = gen_cfg.get('endpoint_random_radius', 0.3)
    multiplier = gen_cfg.get('multiplier', 3)
    reward_target_min = gen_cfg.get('reward_target_min', 0.9)
    reward_target_max = gen_cfg.get('reward_target_max', 1.0)
    max_radius_ratio = max(0.0, min(1.0, 1.0 - reward_target_min))

    # Tasks configuration: multiple input folders with per-task max_trajectories
    tasks_cfg = config.get('tasks', {})
    tasks = []
    for tname, t in tasks_cfg.items():
        tpath = t.get('path')
        tmax = t.get('max_trajectories', None)
        # Normalize null to None and ensure integers when provided
        if tmax is not None:
            try:
                tmax = int(tmax)
                if tmax < 0:
                    tmax = None
            except Exception:
                tmax = None
        if not tpath:
            print(f"Warning: task {tname} has no path, skipping")
            continue
        tasks.append({'name': tname, 'path': tpath, 'max_trajectories': tmax})

    if not tasks:
        raise ValueError('No valid tasks found in config/tasks')

    # Update stats with configuration
    stats['curve_type_used'] = curve_type
    stats['chunk_size_used'] = chunk_size

    episode_cnt = 0
    len_min = 10000
    episode_id = 0

    # Iterate over tasks
    for t in tasks:
        root_path = t['path']
        max_trajectories = t['max_trajectories']

        if not os.path.exists(root_path):
            print(f"Warning: root directory for task {t['name']} not found: {root_path}. Skipping task.")
            continue

        h5_path = os.path.join(root_path, 'sample_points.h5')
        if not os.path.exists(h5_path):
            print(f"Warning: {h5_path} does not exist for task {t['name']}. Skipping task.")
            continue

        print(f"Processing task {t['name']} -> {h5_path} (quota: {max_trajectories})...")

        with h5py.File(h5_path, 'r') as h5_data:
            endpoint = h5_data['endpoint'][6:12]
            eef_positions = np.array(h5_data['state/eef/position'])
            stats['total_eef_positions'] += len(eef_positions)
            # initialize per-task stats
            per_task_stats[t['name']] = {
                'total_eef_positions': len(eef_positions),
                'items_to_process': 0,
                'successful_episodes': 0,
                'skipped_episodes': 0,
                'missing_image_paths': 0,
                'total_images_processed': 0,
                'curve_lengths': [],
                'episodes_with_truncated_curves': 0,
                'episodes_with_extended_curves': 0,
            }

        if not args.reward:
            if max_trajectories is not None and max_trajectories > 0:
                items_to_process = min(max_trajectories, len(eef_positions))
                print(f"  Limiting trajectories to: {items_to_process} (task quota)")
            else:
                items_to_process = len(eef_positions)
                print(f"  Processing all trajectories: {items_to_process}")
            per_task_stats[t['name']]['items_to_process'] = items_to_process
            sampled_indices = np.random.permutation(len(eef_positions))[:items_to_process]
            pbar = tqdm(total=items_to_process, desc=f"Processing {t['name']}")
            for eef_id in sampled_indices:
                eef_position = eef_positions[eef_id]
                xyz_start = eef_position[6:9]
                xyz_end = endpoint[0:3]
                rpy_start = eef_position[9:12]
                rpy_end = endpoint[3:6]
                curve = generate_curve(curve_type, xyz_start, xyz_end, rpy_end, beta)
                curve_length = len(curve)
                if curve_length == 0:
                    curve = np.vstack([curve, xyz_end[np.newaxis, :]])
                    curve_length = 1
                per_task_stats[t['name']]['curve_lengths'].append(curve_length)
                if curve_length < len_min:
                    len_min = curve_length
                    episode_id = eef_id
                gripper = np.tile([0.0, 0.0], (curve_length, 1))
                if curve_length < chunk_size:
                    per_task_stats[t['name']]['episodes_with_extended_curves'] += 1
                    gripper[-1] = [0.0, 90.0]
                    while len(curve) < chunk_size:
                        curve = np.vstack([curve, curve[-1]])
                        gripper = np.vstack([gripper, gripper[-1]])
                    curve_length = len(curve)
                elif curve_length > chunk_size:
                    curve = curve[:chunk_size]
                    gripper = gripper[:chunk_size]
                    curve_length = chunk_size
                    per_task_stats[t['name']]['episodes_with_truncated_curves'] += 1
                rpy_state = generate_rpy_trajectory(rpy_start, rpy_end, curve_length)
                left_curve = np.tile(eef_position[0:3], (curve_length, 1))
                left_rpy = np.tile(eef_position[3:6], (curve_length, 1))
                combined_data = np.hstack((left_curve, left_rpy, curve, rpy_state))
                img_path = os.path.join(root_path, 'camera', str(eef_id))
                if not os.path.exists(img_path):
                    per_task_stats[t['name']]['missing_image_paths'] += 1
                    per_task_stats[t['name']]['skipped_episodes'] += 1
                    stats['total_missing_image_paths'] += 1
                    print(f"Warning: Image path {img_path} does not exist. Skipping eef_id {eef_id}.")
                    continue
                img_files = [img for img in os.listdir(img_path) if img.endswith('.jpg')]
                stats['total_images_processed'] += len(img_files)
                per_task_stats[t['name']]['total_images_processed'] += len(img_files)
                imgs = {}
                for img_file in img_files:
                    img_name = os.path.splitext(img_file)[0]
                    img_full_path = os.path.join(img_path, img_file)
                    imgs[img_name] = Image.open(img_full_path)
                generate_episode(output_path, episode_cnt, combined_data, gripper, imgs, reward_value=None)
                episode_cnt += 1
                per_task_stats[t['name']]['successful_episodes'] += 1
                stats['total_successful_episodes'] += 1
                pbar.update(1)
            pbar.close()
        else:
            if max_trajectories is not None and max_trajectories > 0:
                base_items = min(max_trajectories, len(eef_positions))
                print(f"  Limiting base trajectories: {base_items} (task quota)")
            else:
                base_items = len(eef_positions)
                print(f"  Processing all base trajectories: {base_items}")
            items_to_process = base_items * multiplier
            per_task_stats[t['name']]['items_to_process'] = items_to_process
            print(f"  Applying multiplier={multiplier} -> planning to generate {items_to_process} episodes")
            base_indices = np.random.permutation(len(eef_positions))[:base_items]
            pbar = tqdm(total=items_to_process, desc=f"Processing {t['name']} [reward]")
            for eef_id in base_indices:
                eef_position = eef_positions[eef_id]
                xyz_start = eef_position[6:9]
                original_endpoint_pos = endpoint[0:3]
                rpy_start = eef_position[9:12]
                rpy_end = endpoint[3:6]
                R_radius = endpoint_random_radius
                if multiplier > 1:
                    radii = np.linspace(0.0, max_radius_ratio * R_radius, multiplier)
                else:
                    radii = np.array([0.0])
                for radius in radii:
                    rand_dir = np.random.normal(size=3)
                    base_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-9)
                    random_endpoint_pos = original_endpoint_pos - base_dir * radius
                    dist_random = np.linalg.norm(original_endpoint_pos - random_endpoint_pos)
                    if dist_random < 1e-9 or R_radius < 1e-7:
                        reward_value = 1.0
                    else:
                        reward_value = 1.0 - (dist_random / (R_radius + 1e-12))
                    reward_value = min(max(reward_value, reward_target_min), reward_target_max)
                    xyz_end = random_endpoint_pos
                    curve = generate_curve(curve_type, xyz_start, xyz_end, rpy_end, beta)
                    curve_length = len(curve)
                    if curve_length == 0:
                        curve = np.vstack([curve, xyz_end[np.newaxis, :]])
                        curve_length = 1
                    per_task_stats[t['name']]['curve_lengths'].append(curve_length)
                    if curve_length < len_min:
                        len_min = curve_length
                        episode_id = eef_id
                    gripper = np.tile([0.0, 0.0], (curve_length, 1))
                    if curve_length < chunk_size:
                        per_task_stats[t['name']]['episodes_with_extended_curves'] += 1
                        gripper[-1] = [0.0, 90.0]
                        while len(curve) < chunk_size:
                            curve = np.vstack([curve, curve[-1]])
                            gripper = np.vstack([gripper, gripper[-1]])
                        curve_length = len(curve)
                    elif curve_length > chunk_size:
                        curve = curve[:chunk_size]
                        gripper = gripper[:chunk_size]
                        curve_length = chunk_size
                        per_task_stats[t['name']]['episodes_with_truncated_curves'] += 1
                    rpy_state = generate_rpy_trajectory(rpy_start, rpy_end, curve_length)
                    left_curve = np.tile(eef_position[0:3], (curve_length, 1))
                    left_rpy = np.tile(eef_position[3:6], (curve_length, 1))
                    combined_data = np.hstack((left_curve, left_rpy, curve, rpy_state))
                    per_task_stats[t['name']].setdefault('reward_means', []).append(reward_value)
                    per_task_stats[t['name']].setdefault('reward_mins', []).append(reward_value)
                    per_task_stats[t['name']].setdefault('reward_maxs', []).append(reward_value)
                    stats.setdefault('reward_all', []).append(reward_value)
                    img_path = os.path.join(root_path, 'camera', str(eef_id))
                    if not os.path.exists(img_path):
                        per_task_stats[t['name']]['missing_image_paths'] += 1
                        per_task_stats[t['name']]['skipped_episodes'] += 1
                        stats['total_missing_image_paths'] += 1
                        print(f"Warning: Image path {img_path} does not exist. Skipping eef_id {eef_id}.")
                        continue
                    img_files = [img for img in os.listdir(img_path) if img.endswith('.jpg')]
                    stats['total_images_processed'] += len(img_files)
                    per_task_stats[t['name']]['total_images_processed'] += len(img_files)
                    imgs = {}
                    for img_file in img_files:
                        img_name = os.path.splitext(img_file)[0]
                        img_full_path = os.path.join(img_path, img_file)
                        imgs[img_name] = Image.open(img_full_path)
                    generate_episode(output_path, episode_cnt, combined_data, gripper, imgs, reward_value=reward_value)
                    episode_cnt += 1
                    per_task_stats[t['name']]['successful_episodes'] += 1
                    stats['total_successful_episodes'] += 1
                    pbar.update(1)
            pbar.close()
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print simplified global statistics and per-task summaries
    print("\n" + "="*60)
    print("           Processing Complete - Summary")
    print("="*60)
    print(f"Processing time: {processing_time:.2f} s")
    print(f"Curve type used: {stats['curve_type_used']}")
    print(f"Chunk size: {stats['chunk_size_used']}")
    if args.reward:
        print(f"Reward mode: multiplier={multiplier} radius={endpoint_random_radius} range=[{reward_target_min},{reward_target_max}]")
    print()

    print("Global statistics:")
    print(f"  Total EEF positions: {stats['total_eef_positions']}")
    if 'max_trajectories' in locals() and max_trajectories is not None and max_trajectories > 0:
        print(f"  Global trajectory cap (config): {max_trajectories}")
    print(f"  Total successful episodes: {stats['total_successful_episodes']}")
    print(f"  Total images processed: {stats['total_images_processed']}")
    print(f"  Missing / inaccessible image paths: {stats['total_missing_image_paths']}")
    print()

    print("Per-task summary:")
    for tname, tstats in per_task_stats.items():
        print(f"  Task: {tname}")
        print(f"    EEF point count: {tstats['total_eef_positions']}")
        print(f"    Planned episodes: {tstats['items_to_process']}")
        print(f"    Successful episodes: {tstats['successful_episodes']}")
        print(f"    Skipped episodes: {tstats['skipped_episodes']}")
        print(f"    Missing image paths: {tstats['missing_image_paths']}")
        print(f"    Images processed: {tstats['total_images_processed']}")
        if tstats['curve_lengths']:
            avg_len = np.mean(tstats['curve_lengths'])
            std_len = np.std(tstats['curve_lengths'])
            print(f"    Mean curve length: {avg_len:.2f} ± {std_len:.2f}")
            print(f"    Curves truncated: {tstats['episodes_with_truncated_curves']}")
            print(f"    Curves extended: {tstats['episodes_with_extended_curves']}")
        if args.reward and tstats.get('reward_means'):
            r_task_mean = np.mean(tstats['reward_means'])
            r_task_std = np.std(tstats['reward_means'])
            r_task_min = np.min(tstats['reward_mins'])
            r_task_max = np.max(tstats['reward_maxs'])
            print(f"    Reward distribution: mean={r_task_mean:.4f} ± {r_task_std:.4f} min={r_task_min:.4f} max={r_task_max:.4f}")
        print()

    print("Done. Refer to the per-task summaries above.")

if __name__ == "__main__":
    main()
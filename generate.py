import os
import yaml
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import time

from utils.visualize_points import visualize_curve_with_rpy
from utils.bezier_util import generate_bezier_trajectory
from utils.cone_util import generate_cone_trajectory
from utils.rpy_util import generate_rpy_trajectory
from scipy.spatial.transform import Rotation as R

def curve_length(curve):
    return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))

def get_direct(end, rpy_end):
    """根据末端位姿 rpy_end 计算其在世界坐标系中的朝向(取本地Z轴)的反方向单位向量。

    参数:
        end: ndarray(shape=(3,)) 末端位置(此处不参与计算, 预留接口)
        rpy_end: ndarray(shape=(3,)) 末端的 (roll, pitch, yaw)
    返回:
        ndarray(shape=(3,)) 反方向单位向量
    说明:
        这里约定前向方向使用末端坐标系的 Z 轴 (0,0,1) 经过 RPY 旋转后的结果。
        如果项目实际前向轴不同(例如 X 轴), 可在此处修改基向量。
    """
    rpy_end = np.asarray(rpy_end, dtype=float)
    if rpy_end.shape != (3,):
        raise ValueError("rpy_end 必须是长度为3的一维数组")
    rot = R.from_euler('xyz', rpy_end)
    forward = rot.apply(np.array([0.0, 0.0, 1.0]))  # 本地Z轴
    vec = -forward
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def generate_curve(curve_type, start, end, rpy_end, beta, maxn = 2000):
    direct = get_direct(end, rpy_end)
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

def generate_episode(output_dir, episode_id, combined_data, gripper, imgs):
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

        # Save the combined data into the position dataset
        g_action_eef.create_dataset('position', data=combined_data)
        g_state_eef.create_dataset('position', data=combined_data)

        g_action_gripper.create_dataset('position', data=gripper/90)
        g_state_gripper.create_dataset('position', data=gripper)

    # visualize_curve_with_rpy(combined_data, episode_path)

    # print(f"Episode {episode_id} generated: {episode_path}")

def main():
    # Record start time
    start_time = time.time()
    
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
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    gen_cfg = config.get('generate', {})
    curve_type = gen_cfg.get('curve_type', 'bezier')
    chunk_size = gen_cfg.get('chunk_size', 32)
    output_path = gen_cfg.get('output_path', 'output/')
    beta = gen_cfg.get('beta', 0.003)

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

        # Determine number to process for this task
        if max_trajectories is not None and max_trajectories > 0:
            items_to_process = min(max_trajectories, len(eef_positions))
            print(f"  限制处理轨迹数量: {items_to_process} (任务配额)")
        else:
            items_to_process = len(eef_positions)
            print(f"  处理所有轨迹: {items_to_process}")

        per_task_stats[t['name']]['items_to_process'] = items_to_process

        # Use a while loop to ensure exactly items_to_process valid episodes are generated
        sampled_indices = np.random.permutation(len(eef_positions))
        valid_count = 0
        idx_ptr = 0
        pbar = tqdm(total=items_to_process, desc=f"Processing {t['name']}")
        while valid_count < items_to_process and idx_ptr < len(sampled_indices):
            eef_id = sampled_indices[idx_ptr]
            idx_ptr += 1
            eef_position = eef_positions[eef_id]
            xyz_start = eef_position[6:9]
            xyz_end = endpoint[0:3]

            rpy_start = eef_position[9:12]
            rpy_end = endpoint[3:6]

            # if xyz_start[1] > xyz_end[1]:
            #     per_task_stats[t['name']]['skipped_episodes'] += 1
            #     continue

            # Generate the curve
            curve = generate_curve(curve_type, xyz_start, xyz_end, rpy_end, beta)
            curve_length = len(curve)
            if curve_length == 0:
                curve = np.vstack([curve, xyz_end[np.newaxis, :]])
                curve_length = 1
            # Update per-task curve statistics
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

            # Generate the RPY trajectory with the final curve_length
            rpy_state = generate_rpy_trajectory(rpy_start, rpy_end, curve_length)

            left_curve = np.tile(eef_position[0:3], (curve_length, 1))
            left_rpy = np.tile(eef_position[3:6], (curve_length, 1))

            combined_data = np.hstack((left_curve, left_rpy, curve, rpy_state)
            )

            img_path = os.path.join(root_path, 'camera', str(eef_id))
            if not os.path.exists(img_path):
                per_task_stats[t['name']]['missing_image_paths'] += 1
                per_task_stats[t['name']]['skipped_episodes'] += 1
                stats['total_missing_image_paths'] += 1
                print(f"Warning: Image path {img_path} does not exist. Skipping eef_id {eef_id}.")
                continue

            # Load images from the specified path and create a dictionary with image names as keys
            img_files = [img for img in os.listdir(img_path) if img.endswith('.jpg')]
            stats['total_images_processed'] += len(img_files)
            per_task_stats[t['name']]['total_images_processed'] += len(img_files)
            imgs = {}
            for img_file in img_files:
                # Extract image name without extension as key
                img_name = os.path.splitext(img_file)[0]
                img_full_path = os.path.join(img_path, img_file)
                imgs[img_name] = Image.open(img_full_path)

            # Generate the episode
            generate_episode(output_path, episode_cnt, combined_data, gripper, imgs)
            episode_cnt += 1
            per_task_stats[t['name']]['successful_episodes'] += 1
            stats['total_successful_episodes'] += 1
            valid_count += 1
            pbar.update(1)
        pbar.close()
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print simplified global statistics and per-task summaries
    print("\n" + "="*60)
    print("           处理完成 - 简要统计")
    print("="*60)
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"使用的曲线类型: {stats['curve_type_used']}")
    print(f"截断大小: {stats['chunk_size_used']}")
    print()

    print("全局统计:")
    print(f"  总EEF位置数量: {stats['total_eef_positions']}")
    if max_trajectories is not None and max_trajectories > 0:
        print(f"  全局配置限制轨迹数: {max_trajectories}")
    print(f"  成功生成的Episode总数: {stats['total_successful_episodes']}")
    print(f"  处理的图像总数: {stats['total_images_processed']}")
    print(f"  缺失/无法访问图像路径总数: {stats['total_missing_image_paths']}")
    print()

    print("各任务统计摘要:")
    for tname, tstats in per_task_stats.items():
        print(f"  任务: {tname}")
        print(f"    EEF 点数量: {tstats['total_eef_positions']}")
        print(f"    计划处理数量: {tstats['items_to_process']}")
        print(f"    成功生成 Episode: {tstats['successful_episodes']}")
        print(f"    跳过 Episode: {tstats['skipped_episodes']}")
        print(f"    缺失图像路径: {tstats['missing_image_paths']}")
        print(f"    处理图像数: {tstats['total_images_processed']}")
        if tstats['curve_lengths']:
            avg_len = np.mean(tstats['curve_lengths'])
            std_len = np.std(tstats['curve_lengths'])
            print(f"    平均曲线长度: {avg_len:.2f} ± {std_len:.2f}")
            print(f"    被截断曲线数: {tstats['episodes_with_truncated_curves']}")
            print(f"    被扩展曲线数: {tstats['episodes_with_extended_curves']}")
        print()

    print("完成。详见上方每任务摘要。")

if __name__ == "__main__":
    main()
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

def curve_length(curve):
    return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))

def generate_curve(curve_type, start, end, belta = 0.01, maxn = 1000):
    if curve_type == 'bezier':
        curve0 = generate_bezier_trajectory(start, end, num=maxn)
        curve0_length = curve_length(curve0)
        curve = generate_bezier_trajectory(start, end, num=int(curve0_length/belta))
    elif curve_type == 'cone':
        curve0 = generate_cone_trajectory(start, end, num=maxn)
        curve0_length = curve_length(curve0)
        curve = generate_cone_trajectory(start, end, num=int(curve0_length/belta))
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
    
    # Initialize statistics variables
    stats = {
        'total_eef_positions': 0,
        'successful_episodes': 0,
        'skipped_episodes': 0,
        'curve_lengths': [],
        'total_images_processed': 0,
        'missing_image_paths': 0,
        'curve_type_used': '',
        'trunk_size_used': 0,
        'min_curve_length': float('inf'),
        'max_curve_length': 0,
        'episodes_with_truncated_curves': 0,
        'episodes_with_extended_curves': 0
    }
    
    # Load configuration
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    curve_type = config.get('curve_type', 'bezier')
    root_path = config.get('root_path')
    trunk_size = config.get('trunk_size', 32)
    output_path = config.get('output_path', 'output/')
    max_trajectories = config.get('max_trajectories', None)  # Get max trajectories from config
    
    # Update stats with configuration
    stats['curve_type_used'] = curve_type
    stats['trunk_size_used'] = trunk_size

    # Ensure directories exist
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"root directory not found: {root_path}")

    # Process files
    h5_path = os.path.join(root_path, 'sample_points.h5')
    print(f"Processing {h5_path}...")

    with h5py.File(h5_path, 'r') as h5_data:
        # Call the appropriate curve generation function
        endpoint = h5_data['endpoint'][6:12]
        eef_positions = np.array(h5_data['state/eef/position'])
        stats['total_eef_positions'] = len(eef_positions)
    
    episode_cnt = 0
    len_min = 10000
    episode_id = 0
    
    # Determine the number of items to process based on config
    if max_trajectories is not None and max_trajectories > 0:
        items_to_process = min(max_trajectories, len(eef_positions))
        print(f"限制处理轨迹数量: {items_to_process} (来自配置文件)")
    else:
        items_to_process = len(eef_positions)
        print(f"处理所有轨迹: {items_to_process}")
    
    # Add tqdm progress bar to the loop
    for eef_id, eef_position in enumerate(tqdm(eef_positions[:items_to_process], desc="Processing eef_positions")):
        xyz_start = eef_position[6:9]
        xyz_end = endpoint[0:3]
        # Generate the curve
        curve = generate_curve(curve_type, xyz_start, xyz_end)
        curve_length = len(curve)
        
        # Update curve length statistics
        stats['curve_lengths'].append(curve_length)
        stats['min_curve_length'] = min(stats['min_curve_length'], curve_length)
        stats['max_curve_length'] = max(stats['max_curve_length'], curve_length)

        if curve_length < len_min:
            len_min = curve_length
            episode_id = eef_id

        gripper = np.tile([0.0, 0.0], (curve_length, 1))
        if curve_length < trunk_size:
            stats['episodes_with_extended_curves'] += 1
            gripper[-1] = [0.0, 90.0]
            while len(curve) < trunk_size:
                curve = np.vstack([curve, curve[-1]])
                gripper = np.vstack([gripper, gripper[-1]])
            curve_length = len(curve)
        elif curve_length > trunk_size:
            curve = curve[:trunk_size]
            gripper = gripper[:trunk_size]
            curve_length = trunk_size
            stats['episodes_with_truncated_curves'] += 1

        rpy_start = eef_position[9:12]
        rpy_end = endpoint[3:6]
        # Generate the RPY trajectory with the final curve_length
        rpy_state = generate_rpy_trajectory(rpy_start, rpy_end, curve_length)

        left_curve = np.tile(eef_position[0:3], (curve_length, 1))
        left_rpy = np.tile(eef_position[3:6], (curve_length, 1))

        combined_data = np.hstack((left_curve, left_rpy, curve, rpy_state))

        img_path = os.path.join(root_path, 'camera', str(eef_id))
        if not os.path.exists(img_path):
            stats['missing_image_paths'] += 1
            stats['skipped_episodes'] += 1
            print(f"Warning: Image path {img_path} does not exist. Break at eef_id {eef_id}.")
            break

        # Load images from the specified path and create a dictionary with image names as keys
        img_files = [img for img in os.listdir(img_path) if img.endswith('.jpg')]
        stats['total_images_processed'] += len(img_files)
        imgs = {}
        for img_file in img_files:
            # Extract image name without extension as key
            img_name = os.path.splitext(img_file)[0]
            img_full_path = os.path.join(img_path, img_file)
            imgs[img_name] = Image.open(img_full_path)

        # Generate the episode
        generate_episode(output_path, episode_cnt, combined_data, gripper, imgs)
        episode_cnt += 1
        stats['successful_episodes'] += 1
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("           处理完成 - 统计信息")
    print("="*60)
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"使用的曲线类型: {stats['curve_type_used']}")
    print(f"截断大小: {stats['trunk_size_used']}")
    print()
    
    print("数据处理统计:")
    print(f"  总EEF位置数量: {stats['total_eef_positions']}")
    if max_trajectories is not None and max_trajectories > 0:
        print(f"  配置限制处理数量: {items_to_process}")
    print(f"  成功生成的Episode数量: {stats['successful_episodes']}")
    print(f"  跳过的Episode数量: {stats['skipped_episodes']}")
    print(f"  缺失图像路径数量: {stats['missing_image_paths']}")
    print(f"  处理的图像总数: {stats['total_images_processed']}")
    print()
    
    print("曲线统计:")
    if stats['curve_lengths']:
        avg_curve_length = np.mean(stats['curve_lengths'])
        std_curve_length = np.std(stats['curve_lengths'])
        print(f"  最小曲线长度: {stats['min_curve_length']}")
        print(f"  最大曲线长度: {stats['max_curve_length']}")
        print(f"  平均曲线长度: {avg_curve_length:.2f}")
        print(f"  曲线长度标准差: {std_curve_length:.2f}")
        print(f"  被截断的曲线数量: {stats['episodes_with_truncated_curves']}")
        print(f"  被扩展的曲线数量: {stats['episodes_with_extended_curves']}")
    print()
    
    print("成功率统计:")
    if items_to_process > 0:
        success_rate = (stats['successful_episodes'] / items_to_process) * 100
        print(f"  Episode成功生成率: {success_rate:.2f}%")
        
        if stats['successful_episodes'] > 0:
            avg_images_per_episode = stats['total_images_processed'] / stats['successful_episodes']
            print(f"  每个Episode平均图像数: {avg_images_per_episode:.2f}")
    print()
    
    print("原始统计信息:")
    print(f"  最小长度: {len_min}, 对应Episode ID: {episode_id}")
    print("="*60)

if __name__ == "__main__":
    main()
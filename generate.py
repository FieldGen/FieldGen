import os
import yaml
import h5py
import numpy as np
from PIL import Image
from utils.bezier_util import generate_bezier_trajectory
from utils.cone_util import generate_cone_trajectory
from utils.cone_util import generate_rpy_trajectory

def curve_length(curve):
    return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))

def generate_curve(curve_type, start, end, trunk_size, belta = 0.003):
    if curve_type == 'bezier':
        curve0 = generate_bezier_trajectory(start, end, num=2000)
        curve0_length = curve_length(curve0)
        curve = generate_bezier_trajectory(start, end, num=int(curve0_length/belta))
    elif curve_type == 'cone':
        curve0 = generate_cone_trajectory(start, end, num=2000)
        curve0_length = curve_length(curve0)
        curve = generate_cone_trajectory(start, end, num=int(curve0_length/belta))
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}")

    if len(curve) > trunk_size:
        curve = curve[:trunk_size]
    
    return curve

def generate_episode(output_dir, episode_id, curve, rpy_state, imgs):
    episode_path = os.path.join(output_dir, f"episode{episode_id}")
    os.makedirs(episode_path, exist_ok=True)

    # Ensure camera/0 directory exists
    camera_path = os.path.join(episode_path, 'camera', '0')
    os.makedirs(camera_path, exist_ok=True)
    # Save the img
    hand_left_image_path = os.path.join(camera_path, 'hand_left.jpg')
    hand_right_image_path = os.path.join(camera_path, 'hand_right.jpg')
    head_image_path = os.path.join(camera_path, 'head.jpg')
    imgs[0].save(hand_left_image_path)
    imgs[1].save(hand_right_image_path)
    imgs[2].save(head_image_path)

    h5_path = os.path.join(episode_path, 'aligned_joints.h5')
    with h5py.File(h5_path, 'w') as h5_file:
        h5_file.create_dataset('timestamps', data=np.arange(len(curve)))
        g_action = h5_file.create_group('action')
        g_state = h5_file.create_group('state')
        g_action_eef = g_action.create_group('eef')
        g_state_eef = g_state.create_group('eef')
        
        # Combine curve and rpy_state into a single array of shape (T, 6)
        combined_data = np.hstack((curve, rpy_state))

        # Save the combined data into the position dataset
        g_action_eef.create_dataset('position', data=combined_data)
        g_state_eef.create_dataset('position', data=combined_data)

        g_effector = h5_file.create_group('effector')
        g_effector.create_dataset('position', data=curve)

    print(f"Episode {episode_id} generated: {episode_path}")

def main():
    # Load configuration
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    curve_type = config.get('curve_type', 'bezier')
    root_path = config.get('root_path')
    trunk_size = config.get('trunk_size', 32)

    # Ensure directories exist
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"root directory not found: {root_path}")

    # Process files
    h5_path = os.path.join(root_path, 'sample_points.h5')
    print(f"Processing {h5_path}...")

    with h5py.File(h5_path, 'r') as h5_data:
        # Call the appropriate curve generation function
        endpoint = h5_data['endpoints'][6:12]
        eef_positions = np.array(h5_data['state/eef/position'])
    
    for eef_id, eef_position in enumerate(eef_positions):
        xyz_start = eef_position[6:9]
        xyz_end = endpoint[0:3]
        # Generate the curve
        curve = generate_curve(curve_type, xyz_start, xyz_end, trunk_size)
        curve_length = len(curve)

        rpy_start = eef_position[9:12]
        rpy_end = endpoint[3:6]
        # Generate the RPY trajectory
        rpy_state = generate_rpy_trajectory(rpy_start, rpy_end, curve_length)

        img_path = os.path.join(root_path, 'camera', eef_id)
        imgs = [img for img in os.listdir(img_path) if img.endswith('.jpg')]

        # Generate the episode
        generate_episode(root_path, 0, curve, rpy_state, imgs)





if __name__ == "__main__":
    main()
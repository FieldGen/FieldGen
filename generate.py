import os
import yaml
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.bezier_util import generate_bezier_trajectory
from utils.cone_util import generate_cone_trajectory
from utils.rpy_util import generate_rpy_trajectory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

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

    visualize_curve_with_rpy(curve, rpy_state, episode_path)

    # print(f"Episode {episode_id} generated: {episode_path}")

def visualize_curve_with_rpy(curve, rpy_state, episode_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the curve
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label='Curve', color='blue')

    # Add start and end points
    ax.scatter(curve[0, 0], curve[0, 1], curve[0, 2], color='green', label='Start', s=50)
    ax.scatter(curve[-1, 0], curve[-1, 1], curve[-1, 2], color='orange', label='End', s=50)

    # Determine dynamic length based on y-axis scale
    y_range = ax.get_ylim()
    dynamic_length = (y_range[1] - y_range[0]) * 0.2  # 1% of y-axis range

    # Visualize rpy as rotation directions
    for i in range(len(curve)):
        rotation = R.from_euler('xyz', rpy_state[i])
        direction = rotation.apply([0, 0, 1])  # Use y-axis as reference direction
        ax.quiver(
            curve[i, 0], curve[i, 1], curve[i, 2],
            direction[0], direction[1], direction[2],
            length=dynamic_length, color='red', arrow_length_ratio=0.3
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Save the plot
    plot_path = os.path.join(episode_path, 'curve_visualization.png')
    plt.savefig(plot_path)
    plt.close()

def main():
    # Load configuration
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    curve_type = config.get('curve_type', 'bezier')
    root_path = config.get('root_path')
    trunk_size = config.get('trunk_size', 32)
    output_path = config.get('output_path', 'output/')

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
    
    episode_cnt = 0
    # Add tqdm progress bar to the loop
    for eef_id, eef_position in enumerate(tqdm(eef_positions, desc="Processing eef_positions")):
        xyz_start = eef_position[6:9]
        xyz_end = endpoint[0:3]
        # Generate the curve
        curve = generate_curve(curve_type, xyz_start, xyz_end, trunk_size)
        curve_length = len(curve)

        if curve_length < 10:
            print(f"Warning: No valid curve generated for eef_id {eef_id}. Skipping.")
            continue

        rpy_start = eef_position[9:12]
        rpy_end = endpoint[3:6]
        # Generate the RPY trajectory
        rpy_state = generate_rpy_trajectory(rpy_start, rpy_end, curve_length)

        img_path = os.path.join(root_path, 'camera', str(eef_id))
        if not os.path.exists(img_path):
            print(f"Warning: Image path {img_path} does not exist. Break at eef_id {eef_id}.")
            break

            # Load images from the specified path
        imgs = [img for img in os.listdir(img_path) if img.endswith('.jpg')]
        # Prepend the full path to the image filenames
        imgs = [os.path.join(img_path, img) for img in imgs]
        imgs = [Image.open(img) for img in imgs]

        # Generate the episode
        generate_episode(output_path, episode_cnt, curve, rpy_state, imgs)
        episode_cnt += 1

if __name__ == "__main__":
    main()
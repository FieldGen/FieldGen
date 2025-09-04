import os
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R

def visualize_curve_with_rpy(combined_data, episode_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    curve = combined_data[:, 6:9]  # Extract the curve points
    rpy_state = combined_data[:, 9:12]  # Extract the RPY

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

def create_3d_scatter_plot(points_xyz, endpoint_xyz):
    """
    Creates an interactive 3D scatter plot for the given points and endpoint.
    
    Args:
        points_xyz (np.array): Array of XYZ coordinates for the main points.
        endpoint_xyz (np.array): Array with the XYZ coordinates for the endpoint.
        
    Returns:
        go.Figure: A Plotly figure object for the 3D scatter plot.
    """
    fig = go.Figure(data=[
        # All points scatter plot
        go.Scatter3d(
            x=points_xyz[:, 0],
            y=points_xyz[:, 1],
            z=points_xyz[:, 2],
            mode='markers',
            name='数据点',
            marker=dict(
                size=2,
                color=points_xyz[:, 2],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Z 坐标')
            )
        ),
        # Highlighted Endpoint
        go.Scatter3d(
            x=[endpoint_xyz[0]],
            y=[endpoint_xyz[1]],
            z=[endpoint_xyz[2]],
            mode='markers',
            name='Endpoint',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                opacity=1
            )
        )
    ])

    fig.update_layout(
        title='三维空间点云分布（含 Endpoint）',
        scene=dict(
            xaxis_title='X 轴',
            yaxis_title='Y 轴',
            zaxis_title='Z 轴'
        ),
        margin=dict(r=20, b=10, l=10, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def create_2d_density_plots(points_xyz):
    """
    Creates 2D density heatmaps for XY, XZ, and YZ views.
    
    Args:
        points_xyz (np.array): Array of XYZ coordinates for the points.
        
    Returns:
        go.Figure: A Plotly figure object for the 2D density plots.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('XY 平面密度', 'XZ 平面密度', 'YZ 平面密度')
    )

    # XY-plane density
    fig.add_trace(
        go.Histogram2d(
            x=points_xyz[:, 0],
            y=points_xyz[:, 1],
            colorscale='Viridis',
            name='XY'
        ),
        row=1, col=1
    )

    # XZ-plane density
    fig.add_trace(
        go.Histogram2d(
            x=points_xyz[:, 0],
            y=points_xyz[:, 2],
            colorscale='Viridis',
            name='XZ'
        ),
        row=1, col=2
    )

    # YZ-plane density
    fig.add_trace(
        go.Histogram2d(
            x=points_xyz[:, 1],
            y=points_xyz[:, 2],
            colorscale='Viridis',
            name='YZ'
        ),
        row=1, col=3
    )

    fig.update_layout(
        title_text='点云三视图密度图',
        showlegend=False
    )
    fig.update_xaxes(title_text="X 轴", row=1, col=1)
    fig.update_yaxes(title_text="Y 轴", row=1, col=1)
    fig.update_xaxes(title_text="X 轴", row=1, col=2)
    fig.update_yaxes(title_text="Z 轴", row=1, col=2)
    fig.update_xaxes(title_text="Y 轴", row=1, col=3)
    fig.update_yaxes(title_text="Z 轴", row=1, col=3)
    
    return fig

def visualize_point_distribution():
    """
    Loads endpoint data from an H5 file and visualizes the 3D point distribution
    and 2D density maps.
    """
    # Load configuration to find task data paths (support multiple tasks)
    config_path = os.path.join('config', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"错误：配置文件未找到于 '{config_path}'")
        print("请确保配置文件存在或从 config.yaml.template 创建。")
        return

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    tasks_cfg = config.get('tasks')

    task_list = []
    # If tasks specified, use them; otherwise fall back to single root_path for backward compatibility
    if tasks_cfg:
        for tname, t in tasks_cfg.items():
            tpath = t.get('path')
            if not tpath:
                print(f"警告：task {tname} 未配置 path，已跳过。")
                continue
            task_list.append({'name': tname, 'path': tpath})
    else:
        root_path = config.get('root_path')
        if not root_path:
            print("错误：config 中既没有 'tasks' 也没有 'root_path'。请配置数据路径。")
            return
        task_list.append({'name': 'default', 'path': root_path})

    any_processed = False
    for t in task_list:
        root_path = t['path']
        tname = t['name']

        if not os.path.exists(root_path):
            print(f"警告：任务 {tname} 的路径不存在: {root_path}，已跳过。")
            continue

        h5_path = os.path.join(root_path, 'sample_points.h5')
        if not os.path.exists(h5_path):
            print(f"警告：数据文件未找到于: {h5_path}（任务 {tname}），已跳过。")
            continue

        print(f"正在处理任务 {tname} 的文件: {h5_path}...")

        with h5py.File(h5_path, 'r') as h5_data:
            if 'state/eef/position' not in h5_data:
                print(f"错误：在 H5 文件中未找到 'state/eef/position' 数据集（任务 {tname}）。")
                continue
            if 'endpoint' not in h5_data:
                print(f"错误：在 H5 文件中未找到 'endpoint' 数据集（任务 {tname}）。")
                continue

            eef_positions = np.array(h5_data['state/eef/position'])
            endpoint_data = np.array(h5_data['endpoint'])

        if eef_positions.shape[1] < 9:
            print(f"错误：数据点维度不足（任务 {tname}）。需要至少9个维度，但只有 {eef_positions.shape[1]}。")
            continue

        points_xyz = eef_positions[:, 6:9]
        endpoint_xyz = endpoint_data[6:9]

        print(f"已加载 {len(points_xyz)} 个点（任务 {tname}）。正在生成可视化图表...")

        # Generate and save 3D scatter plot
        fig_3d = create_3d_scatter_plot(points_xyz, endpoint_xyz)
        output_3d_filename = f'html/point_distribution_3d_{tname}.html'
        fig_3d.write_html(output_3d_filename)

        # Generate and save 2D density plots
        fig_2d = create_2d_density_plots(points_xyz)
        output_2d_filename = f'html/point_distribution_2d_density_{tname}.html'
        fig_2d.write_html(output_2d_filename)

        print("\n" + "="*60)
        print(f"任务 {tname} 的可视化完成")
        print("="*60)
        print(f"1. 交互式三维散点图已保存至: {os.path.abspath(output_3d_filename)}")
        print(f"2. 三视图密度图已保存至: {os.path.abspath(output_2d_filename)}")
        print("请在您的浏览器中打开这些 HTML 文件以查看。")
        print("="*60 + "\n")

        any_processed = True

    if not any_processed:
        print("未处理任何任务。请检查配置文件和数据路径。")


if __name__ == "__main__":
    visualize_point_distribution()

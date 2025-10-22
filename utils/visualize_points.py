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

    curve = combined_data[:, 6:9]  # Curve points
    rpy_state = combined_data[:, 9:12]  # RPY orientation

    # Plot polyline
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label='Curve', color='blue')

    # Start/end markers
    ax.scatter(curve[0, 0], curve[0, 1], curve[0, 2], color='green', label='Start', s=50)
    ax.scatter(curve[-1, 0], curve[-1, 1], curve[-1, 2], color='orange', label='End', s=50)

    # Dynamic arrow length scale
    y_range = ax.get_ylim()
    dynamic_length = (y_range[1] - y_range[0]) * 0.2  # 1% of y-axis range

    # Orientation arrows
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

    # Save figure
    plot_path = os.path.join(episode_path, 'curve_visualization.png')
    plt.savefig(plot_path)
    plt.close()

def create_3d_scatter_plot(points_xyz, endpoint_xyz):
    """Interactive 3D scatter with highlighted endpoint."""
    fig = go.Figure(data=[
        # All points scatter plot
        go.Scatter3d(
            x=points_xyz[:, 0],
            y=points_xyz[:, 1],
            z=points_xyz[:, 2],
            mode='markers',
            name='points',
            marker=dict(
                size=2,
                color=points_xyz[:, 2],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Z')
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
    title='3D Point Distribution (with Endpoint)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=20, b=10, l=10, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def create_2d_density_plots(points_xyz):
    """2D density heatmaps for XY / XZ / YZ projections."""
    fig = make_subplots(
        rows=1, cols=3,
    subplot_titles=('XY Density', 'XZ Density', 'YZ Density')
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

    fig.update_layout(title_text='Multi-view Density', showlegend=False)
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Z", row=1, col=2)
    fig.update_xaxes(title_text="Y", row=1, col=3)
    fig.update_yaxes(title_text="Z", row=1, col=3)
    
    return fig

def visualize_point_distribution():
    """Render spatial distribution HTML reports for configured tasks.

    Reads each task's `sample_points.h5`, extracts EEF candidate Cartesian points
    and endpoint, and emits (a) a 3D scatter and (b) three 2D density projections.
    """
    config_path = os.path.join('config', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"Error: configuration file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tasks_cfg = config.get('tasks')
    task_list = []
    if tasks_cfg:
        for tname, tdesc in tasks_cfg.items():
            tpath = tdesc.get('path')
            if not tpath:
                print(f"Warning: task {tname} missing path; skipping.")
                continue
            task_list.append({'name': tname, 'path': tpath})
    else:
        root_path = config.get('root_path')
        if not root_path:
            print("Error: neither 'tasks' nor 'root_path' specified in config.")
            return
        task_list.append({'name': 'default', 'path': root_path})

    os.makedirs('html', exist_ok=True)
    any_processed = False
    for t in task_list:
        tname = t['name']
        root_path = t['path']
        if not os.path.exists(root_path):
            print(f"Warning: path missing for task {tname}: {root_path}")
            continue
        h5_path = os.path.join(root_path, 'sample_points.h5')
        if not os.path.exists(h5_path):
            print(f"Warning: sample_points.h5 missing for task {tname}: {h5_path}")
            continue
        print(f"Processing task {tname}: {h5_path}")
        with h5py.File(h5_path, 'r') as h5f:
            if 'state/eef/position' not in h5f or 'endpoint' not in h5f:
                print(f"Error: required datasets missing in {h5_path} (task {tname})")
                continue
            eef_positions = np.array(h5f['state/eef/position'])
            endpoint = np.array(h5f['endpoint'])
        if eef_positions.shape[1] < 9:
            print(f"Error: state/eef/position must have ≥9 columns (task {tname})")
            continue
        points_xyz = eef_positions[:, 6:9]
        endpoint_xyz = endpoint[6:9]
        print(f"  Loaded {len(points_xyz)} points; generating visualizations ...")
        fig3d = create_3d_scatter_plot(points_xyz, endpoint_xyz)
        out3d = f'html/point_distribution_3d_{tname}.html'
        fig3d.write_html(out3d)
        fig2d = create_2d_density_plots(points_xyz)
        out2d = f'html/point_distribution_2d_density_{tname}.html'
        fig2d.write_html(out2d)
        print("\n" + "="*60)
        print(f"Visualization complete for task {tname}")
        print("="*60)
        print(f"1. 3D scatter: {os.path.abspath(out3d)}")
        print(f"2. 2D densities: {os.path.abspath(out2d)}")
        print("Open these HTML files in a browser.")
        print("="*60 + "\n")
        any_processed = True
    if not any_processed:
        print("No tasks processed. Verify configuration paths.")


if __name__ == "__main__":
    visualize_point_distribution()

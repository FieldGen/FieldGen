import os
import argparse
import h5py
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

# 列索引约定 (参考 generate.py 中 combined_data 结构)
# 0:3  -> left arm (占位, 与生成脚本一致)
# 3:6  -> left rpy
# 6:9  -> 右侧末端执行器轨迹 (curve)
# 9:12 -> 右侧末端执行器 rpy 轨迹


def load_episode_points(episode_dir: str, source_group: str = "action", dataset: str = "eef/position"):
    """从单个 episode 目录中读取对齐后的轨迹点。

    Parameters
    ----------
    episode_dir : str
        形如 output/episode0 的目录路径。
    source_group : str
        选择 'action' 或 'state' 组 (两者内容在生成阶段相同)。
    dataset : str
        eef 位置数据集相对路径。

    Returns
    -------
    points : np.ndarray | None
        返回 shape (T, 3) 的末端执行器曲线 (列 6:9)。读取失败返回 None。
    meta : dict
        附加元数据 (episode_name, full_path, timesteps)。
    """
    h5_path = os.path.join(episode_dir, "aligned_joints.h5")
    episode_name = os.path.basename(episode_dir.rstrip(os.sep))
    meta = {"episode_name": episode_name, "h5_path": h5_path}

    if not os.path.isfile(h5_path):
        meta["error"] = "aligned_joints.h5 不存在"
        return None, meta

    try:
        with h5py.File(h5_path, "r") as f:
            ds_path = f"{source_group}/{dataset}"  # e.g. action/eef/position
            if ds_path not in f:
                meta["error"] = f"H5 中不存在数据集: {ds_path}"
                return None, meta
            data = np.array(f[ds_path])  # (T, 12)
            if data.ndim != 2 or data.shape[1] < 9:
                meta["error"] = f"数据维度异常: {data.shape}"
                return None, meta
            curve = data[:, 6:9]
            meta["timesteps"] = curve.shape[0]
            return curve, meta
    except Exception as e:
        meta["error"] = f"读取异常: {e}"
        return None, meta


def plot_single_episode(curve: np.ndarray, title: str):
    """生成单个 episode 的 3D 轨迹图 (Plotly Figure)。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
        mode="lines+markers",
        line=dict(color="#1f77b4", width=4),
        marker=dict(size=3, color=np.linspace(0, 1, len(curve)), colorscale="Viridis", showscale=False),
        name="trajectory"
    ))
    # 起点终点
    fig.add_trace(go.Scatter3d(x=[curve[0, 0]], y=[curve[0, 1]], z=[curve[0, 2]],
                               mode="markers", marker=dict(size=6, color="green"), name="start"))
    fig.add_trace(go.Scatter3d(x=[curve[-1, 0]], y=[curve[-1, 1]], z=[curve[-1, 2]],
                               mode="markers", marker=dict(size=6, color="orange"), name="end"))
    fig.update_layout(title=title, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                      margin=dict(l=10, r=10, b=10, t=40))
    return fig


def plot_all_episodes(output_dir: str,
                      max_episodes: int | None,
                      separate: bool,
                      combined: bool,
                      open_browser: bool,
                      save_dir: str):
    """遍历 output 目录下所有 episode* 文件夹并绘制轨迹。

    Parameters
    ----------
    output_dir : str
        生成脚本输出的根目录 (config.generate.output_path)。
    max_episodes : int | None
        限制处理的 episode 数量 (基于排序后的前 N 个)。None 表示全部。
    separate : bool
        为每个 episode 输出单独 html。
    combined : bool
        生成一个包含所有 episode 的汇总 html。
    open_browser : bool
        是否自动在完成后尝试打开生成的 html (仅在本地环境下有用)。
    """
    if not os.path.isdir(output_dir):
        print(f"错误: 输出目录不存在: {output_dir}")
        return

    # 收集 episode 目录
    episode_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                    if d.startswith("episode") and os.path.isdir(os.path.join(output_dir, d))]
    if not episode_dirs:
        print(f"在 {output_dir} 中未发现 episode* 目录。")
        return

    # 根据数字排序
    def episode_key(p):
        base = os.path.basename(p)
        num = ''.join(filter(str.isdigit, base))
        return int(num) if num else 0

    episode_dirs.sort(key=episode_key)

    if max_episodes is not None and max_episodes > 0:
        episode_dirs = episode_dirs[:max_episodes]

    print(f"将处理 {len(episode_dirs)} 个 episodes。")

    # 准备最终输出目录 (默认项目根目录或用户指定)
    os.makedirs(save_dir, exist_ok=True)

    combined_fig = go.Figure() if combined else None

    for ep_dir in tqdm(episode_dirs, desc="加载轨迹"):
        curve, meta = load_episode_points(ep_dir)
        if curve is None:
            print(f"跳过 {meta['episode_name']}: {meta.get('error')}")
            continue

        ep_name = meta['episode_name']
        # 独立图
        if separate:
            fig = plot_single_episode(curve, title=f"{ep_name} (T={curve.shape[0]})")
            out_file = os.path.join(save_dir, f"{ep_name}_trajectory.html")
            fig.write_html(out_file)
        # 汇总
        if combined_fig is not None:
            combined_fig.add_trace(go.Scatter3d(
                x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
                mode="lines",
                line=dict(width=2),
                name=ep_name,
                hovertext=[f"{ep_name} | t={i}" for i in range(curve.shape[0])],
                hoverinfo="text"
            ))
            # 起终点标记 (可选省略, 为清晰仅用小 marker)
            combined_fig.add_trace(go.Scatter3d(
                x=[curve[0, 0]], y=[curve[0, 1]], z=[curve[0, 2]],
                mode="markers", marker=dict(size=3, color="green"), name=f"{ep_name}_start", showlegend=False
            ))
            combined_fig.add_trace(go.Scatter3d(
                x=[curve[-1, 0]], y=[curve[-1, 1]], z=[curve[-1, 2]],
                mode="markers", marker=dict(size=3, color="orange"), name=f"{ep_name}_end", showlegend=False
            ))

    if combined_fig is not None and len(combined_fig.data) > 0:
        combined_fig.update_layout(
            title=f"所有 Episode 轨迹 (共 {len(episode_dirs)} 个)",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            legend=dict(itemsizing="constant", font=dict(size=10), y=0.99, x=0.01),
            margin=dict(l=10, r=10, b=10, t=40),
        )
        combined_out = os.path.join(save_dir, "all_episodes_trajectories.html")
        combined_fig.write_html(combined_out)
        print(f"汇总 3D 轨迹已保存: {os.path.abspath(combined_out)}")
        if open_browser:
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(combined_out)}")
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="可视化 output 目录中生成的 episode 轨迹 (aligned_joints.h5)")
    parser.add_argument("--output_dir", type=str, default="output", help="生成脚本的输出目录 (含 episode*)")
    parser.add_argument("--max_episodes", type=int, default=None, help="限制处理的 episode 数量")
    parser.add_argument("--no-separate", action="store_true", help="不为每个 episode 生成单独 html")
    parser.add_argument("--no-combined", action="store_true", help="不生成汇总 html")
    parser.add_argument("--open", action="store_true", help="生成后尝试在浏览器打开汇总图")
    parser.add_argument("--save_dir", type=str, default=".", help="输出的 html 保存目录 (默认项目根目录)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_separate and args.no_combined:
        print("已同时禁用单独与汇总输出, 不执行任何绘制。")
        return
    plot_all_episodes(
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        separate=not args.no_separate,
        combined=not args.no_combined,
        open_browser=args.open,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

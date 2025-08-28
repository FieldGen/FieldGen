#!/usr/bin/env python3
"""Generate per-experiment DP performance plots with teleop baseline.

Data come from the CSV-like content the user provided (hard‑coded below).
Each experiment:
  - teleop_success: baseline success rate (%) from teleoperation (first row)
  - dp: list of records with time (min), data volume (chunks/episodes), success (%)

Output: one PNG per experiment under figs/ directory.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go


experiments = {
    "Pick": {
        "teleop_success": 80,
        "teleop_time": 45,  # ≈45
        "teleop_data": "100 episodes",
        "dp": [
            {"time": 3.15, "data": 900, "success": 0},
            {"time": 4.30, "data": 1800, "success": 10},
            {"time": 5.46, "data": 2700, "success": 70},
            {"time": 6.61, "data": 3600, "success": 50},
            {"time": 7.76, "data": 4500, "success": 80},
            {"time": 8.91, "data": 5400, "success": 80},
        ],
    },
    "Rotate(Narrow) Pick": {
        "teleop_success": 100,
        "teleop_time": 45,
        "teleop_data": "100 episodes",
        "dp": [
            {"time": 3.15, "data": 900, "success": 40},
            {"time": 4.30, "data": 1800, "success": 100},
            {"time": 5.46, "data": 2700, "success": 100},
            {"time": 6.61, "data": 3600, "success": 100},
            {"time": 7.76, "data": 4500, "success": 100},
            {"time": 8.91, "data": 5400, "success": 100},
        ],
    },
    "Transparent Pick": {
        "teleop_success": 50,
        "teleop_time": 45,
        "teleop_data": "100 episodes",
        "dp": [
            {"time": 3.15, "data": 900, "success": 0},
            {"time": 4.30, "data": 1800, "success": 70},
            {"time": 5.46, "data": 2700, "success": 80},
            {"time": 6.61, "data": 3600, "success": 100},
            {"time": 7.76, "data": 4500, "success": 100},
            {"time": 8.91, "data": 5400, "success": 100},
        ],
    },
    "Vertical Pick": {
        "teleop_success": 70,
        "teleop_time": 45,
        "teleop_data": "100 episodes",
        "dp": [
            {"time": 3.15, "data": 900, "success": 20},
            {"time": 4.30, "data": 1800, "success": 60},
            {"time": 5.46, "data": 2700, "success": 80},
            {"time": 6.61, "data": 3600, "success": 70},
            {"time": 7.76, "data": 4500, "success": 90},
            {"time": 8.91, "data": 5400, "success": 100},
        ],
    },
    "Special Pick": {
        "teleop_success": 100,
        "teleop_time": 45,
        "teleop_data": "100 episodes",
        "dp": [
            {"time": 3.15, "data": 900, "success": 10},
            {"time": 4.30, "data": 1800, "success": 70},
            {"time": 5.46, "data": 2700, "success": 90},
            {"time": 6.61, "data": 3600, "success": 80},
            {"time": 7.76, "data": 4500, "success": 90},
            {"time": 8.91, "data": 5400, "success": 100},
        ],
    },
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_experiment(name: str, info: dict, out_dir: Path) -> Path:
    """Plot one experiment.

    X 轴仍使用原始浮点时间作为数据点位置（不丢失精度），但刻度标签改成：
        整数分钟\n数据量
    例如： '3\n900' 表示 ≈3 分钟 / 900 chunks。
    """
    teleop = info["teleop_success"]
    records = info["dp"]
    times = [r["time"] for r in records]
    successes = [r["success"] for r in records]

    # 生成多行刻度标签：整数分钟 + 数据量
    tick_labels = [f"{int(round(r['time']))}m\n{r['data']}" for r in records]

    plt.figure(figsize=(5.4, 4.3), dpi=140)
    # DP curve
    plt.plot(times, successes, marker="o", linewidth=2, color="#1f77b4", label="DP")
    for x, y in zip(times, successes):
        plt.text(x, y + 1.5, f"{y}%", ha="center", va="bottom", fontsize=8)

    xmin, xmax = min(times), max(times)
    # Extend small padding for teleop line to look nicer
    pad = (xmax - xmin) * 0.02
    plt.hlines(teleop, xmin - pad, xmax + pad, colors="#d62728", linestyles="--", linewidth=1.8,
               label=f"Teleop {teleop}%")

    plt.xlabel("Time (rounded min) / Data (chunks)")
    plt.ylabel("Success Rate (%)")
    plt.title(name)
    plt.xticks(times, tick_labels)
    plt.ylim(0, 105)
    plt.grid(alpha=0.35, linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{slugify(name)}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_experiment_interactive(name: str, info: dict, out_dir: Path) -> Path:
    """Generate an interactive HTML plot (Plotly) with precise time hover tooltips.

    Hover shows: precise time (float), rounded minute, data volume, success, teleop baseline.
    """
    teleop = info["teleop_success"]
    records = info["dp"]
    times = [r["time"] for r in records]
    rounded_minutes = [int(round(t)) for t in times]
    data_vols = [r["data"] for r in records]
    successes = [r["success"] for r in records]

    fig = go.Figure()
    hovertemplate = (
        f"<b>{name}</b><br>"  # experiment name
        "Time: %{x:.2f} min (≈%{customdata[0]} m)<br>"
        "Data: %{customdata[1]} chunks<br>"
        "Success: %{customdata[2]}%<br>"
        f"Teleop: {teleop}%<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=successes,
            mode="lines+markers",
            name="DP",
            marker=dict(size=8, color="#1f77b4"),
            line=dict(width=2, color="#1f77b4"),
            customdata=list(zip(rounded_minutes, data_vols, successes)),
            hovertemplate=hovertemplate,
        )
    )

    # Teleop baseline line
    fig.add_hline(
        y=teleop,
        line=dict(color="#d62728", width=2, dash="dash"),
        annotation=dict(text=f"Teleop {teleop}%", showarrow=False, yshift=8),
    )

    fig.update_layout(
        title=name,
        xaxis_title="Collect + Process Time (min)",
        yaxis_title="Success Rate (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        margin=dict(l=50, r=20, t=60, b=60),
    )

    # Optional secondary x-axis style labels (rounded minute + data) as tick text
    tickvals = times
    ticktext = [f"{rm}m\n{dv}" for rm, dv in zip(rounded_minutes, data_vols)]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    out_path = out_dir / f"{slugify(name)}_interactive.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def main():
    out_dir = Path("figs")
    out_dir.mkdir(exist_ok=True)
    generated = []
    for name, info in experiments.items():
        static_path = plot_experiment(name, info, out_dir)
        interactive_path = plot_experiment_interactive(name, info, out_dir)
        generated.extend([static_path, interactive_path])
    print("Generated figures (static PNG + interactive HTML):")
    for p in generated:
        print("  ", p)


if __name__ == "__main__":
    main()

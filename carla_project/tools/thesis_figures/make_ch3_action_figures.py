#!/usr/bin/env python3
"""Generate Chapter 3 action-distribution and action-visual-correlation figures."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path("/home/llb/HunyuanWorld-Voyager/bishe")
PROJECT = ROOT / "carla_project"
OUT_DIR = Path(os.environ.get("THESIS_FIG_OUT_DIR", PROJECT / "outputs/figures/thesis_visuals_20260519_compact"))
TOKEN_FILE = PROJECT / "data/tokens_action_corr_f8/tokens_actions.npz"
RAW_ROOT = PROJECT / "data/raw_action_corr_f8"
FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")


def setup_matplotlib() -> font_manager.FontProperties:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        cjk = font_manager.FontProperties(fname=str(FONT_PATH))
    else:
        cjk = font_manager.FontProperties()
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 300
    return cjk


CJK_PROP = None


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        print(f"saved {path}")
    plt.close(fig)


def add_bar_values(ax, bars, fmt="{:.0f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def compute_visual_change(img1: np.ndarray, img2: np.ndarray) -> float:
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    return float((diff > 30).mean())


def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-8:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def lagged_corr(action_signal: np.ndarray, visual_changes: np.ndarray, max_lag: int = 6) -> list[tuple[int, float]]:
    rows = []
    for lag in range(max_lag + 1):
        if lag == 0:
            a = action_signal
            v = visual_changes
        else:
            a = action_signal[:-lag]
            v = visual_changes[lag:]
        if len(a) < 3 or np.std(a) < 1e-8 or np.std(v) < 1e-8:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(a, v)[0, 1])
        rows.append((lag, corr))
    return rows


def make_action_distribution_figure() -> None:
    data = np.load(TOKEN_FILE)
    actions = data["actions"].astype(np.float32)
    steer = actions[:, 0]

    bins = np.linspace(-0.24, 0.24, 25)
    mode_edges = [-1.0, -0.17, -0.06, 0.06, 0.17, 1.0]
    mode_labels = ["强左", "中左", "直行", "中右", "强右"]
    mode_counts = np.histogram(steer, bins=mode_edges)[0]

    csv_path = OUT_DIR / "fig3_3_action_distribution_stats.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "count", "ratio"])
        for label, count in zip(mode_labels, mode_counts):
            writer.writerow([label, int(count), float(count / len(steer))])
    print(f"saved {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    ax = axes[0]
    ax.hist(steer, bins=bins, color="#4c78a8", edgecolor="white", linewidth=0.6)
    ax.axvline(-0.17, color="#888888", linestyle="--", linewidth=1.0)
    ax.axvline(-0.06, color="#888888", linestyle="--", linewidth=1.0)
    ax.axvline(0.06, color="#888888", linestyle="--", linewidth=1.0)
    ax.axvline(0.17, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xlabel("转向角／无量纲", fontproperties=CJK_PROP, fontsize=10)
    ax.set_ylabel("帧数／帧", fontproperties=CJK_PROP, fontsize=10)
    ax.set_title("转向角分布", fontproperties=CJK_PROP, fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    colors = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518", "#b279a2"]
    bars = ax.bar(np.arange(len(mode_labels)), mode_counts, color=colors, width=0.68)
    add_bar_values(ax, bars)
    ax.set_xticks(np.arange(len(mode_labels)))
    ax.set_xticklabels(mode_labels, fontproperties=CJK_PROP, fontsize=9)
    ax.set_xlabel("动作模式／类别", fontproperties=CJK_PROP, fontsize=10)
    ax.set_ylabel("帧数／帧", fontproperties=CJK_PROP, fontsize=10)
    ax.set_title("动作模式统计", fontproperties=CJK_PROP, fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    fig.suptitle("动作分布统计", fontproperties=CJK_PROP, fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.18, wspace=0.26)
    save_figure(fig, "fig3_3_action_distribution")


def load_episode_visual_curve(episode_name: str = "episode_0277") -> tuple[np.ndarray, np.ndarray, list[tuple[int, float]], dict]:
    episode_dir = RAW_ROOT / episode_name
    actions = np.load(episode_dir / "actions.npy").astype(np.float32)
    image_paths = sorted((episode_dir / "images").glob("*.png"))
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    if any(img is None for img in images):
        raise RuntimeError(f"Could not read all images under {episode_dir / 'images'}")

    visual_changes = np.asarray(
        [compute_visual_change(images[i], images[i + 1]) for i in range(len(images) - 1)],
        dtype=np.float32,
    )
    action_signal = (np.abs(actions[:, 0]) * actions[:, 1])[:-1]
    corrs = lagged_corr(action_signal, visual_changes, max_lag=6)
    meta = np.load(episode_dir / "metadata.npy", allow_pickle=True).item()
    return action_signal, visual_changes, corrs, meta


def make_action_visual_curve_figure() -> None:
    action_signal, visual_changes, corrs, meta = load_episode_visual_curve()
    steps = np.arange(len(action_signal))
    u_norm = normalize01(action_signal)
    v_norm = normalize01(visual_changes)
    corr_values = np.asarray([c for _, c in corrs], dtype=np.float32)
    best_idx = int(np.nanargmax(corr_values))
    best_lag, best_corr = corrs[best_idx]

    csv_path = OUT_DIR / "fig3_4_action_visual_curve.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "action_signal_u_t", "visual_change_v_t", "u_norm", "v_norm"])
        for row in zip(steps, action_signal, visual_changes, u_norm, v_norm):
            writer.writerow([int(row[0]), *[float(v) for v in row[1:]]])
        writer.writerow([])
        writer.writerow(["lag", "correlation"])
        for lag, corr in corrs:
            writer.writerow([lag, corr])
    print(f"saved {csv_path}")

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.8), gridspec_kw={"height_ratios": [2.2, 1.0]})

    ax = axes[0]
    ax.plot(steps, u_norm, color="#4c78a8", linewidth=2.1, label="动作强度")
    ax.plot(steps, v_norm, color="#f58518", linewidth=2.1, label="视觉变化")
    ax.set_xlabel("帧序号／帧", fontproperties=CJK_PROP, fontsize=10)
    ax.set_ylabel("归一化幅度／无量纲", fontproperties=CJK_PROP, fontsize=10)
    ax.legend(frameon=False, prop=CJK_PROP, fontsize=9, loc="upper right")
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    lags = np.asarray([lag for lag, _ in corrs])
    ax.plot(lags, corr_values, color="#54a24b", marker="o", linewidth=2.0)
    ax.scatter([best_lag], [best_corr], color="#e45756", zorder=3)
    ax.set_ylim(np.nanmin(corr_values) - 0.10, np.nanmax(corr_values) + 0.12)
    ax.annotate(
        "最优滞后",
        xy=(best_lag, best_corr),
        xytext=(best_lag + 0.45, best_corr - 0.08),
        textcoords="data",
        fontproperties=CJK_PROP,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": "#e45756", "linewidth": 0.8},
        va="center",
    )
    ax.set_xlabel("滞后步数／帧", fontproperties=CJK_PROP, fontsize=10)
    ax.set_ylabel("相关系数／无量纲", fontproperties=CJK_PROP, fontsize=10)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    title = "动作强度与视觉变化曲线（代表性序列）"
    fig.suptitle(title, fontproperties=CJK_PROP, fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.10, hspace=0.34)
    save_figure(fig, "fig3_4_action_visual_change")


def main() -> None:
    global CJK_PROP
    CJK_PROP = setup_matplotlib()
    make_action_distribution_figure()
    make_action_visual_curve_figure()


if __name__ == "__main__":
    main()

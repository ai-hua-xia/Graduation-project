#!/usr/bin/env python3
"""Generate thesis visualization figures from existing CARLA artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import font_manager
from PIL import Image


ROOT = Path("/home/llb/HunyuanWorld-Voyager/bishe")
PROJECT = ROOT / "carla_project"
OUT_DIR = Path(os.environ.get("THESIS_FIG_OUT_DIR", PROJECT / "outputs/figures/thesis_visuals_20260518"))

TOKEN_FILE = PROJECT / "data/tokens_action_corr_f8/tokens_actions.npz"
RAW_ROOT = PROJECT / "data/raw_action_corr_f8"
VQVAE_CKPT = PROJECT / "checkpoints/vqvae/vqvae_action_corr_f8/best.pth"
WM_CKPT = PROJECT / "checkpoints/wm/world_model_f8_adaln_aux_rollout_ft_b1_20260509/best.pth"
ABLATION_CSV = PROJECT / "outputs/evaluations/latest_ablation_20260512/summary.csv"
FINAL_VIDEO = (
    PROJECT
    / "outputs/videos/ablation_demo_eval_sampling_20260518/"
    / "06_adaln_actionaux_rollout_memory_idx105680_evalsampling_100f_controls.mp4"
)
TRAIN_LOG_MAIN = PROJECT / "logs/train_wm/train_world_model_f8_adaln_aux.log"
TRAIN_LOG_ROLLOUT = PROJECT / "logs/train_wm/train_world_model_f8_adaln_aux_rollout_ft_b1_20260509.log"
VQVAE_TRAIN_LOG = PROJECT / "logs/train_vqvae/train_vqvae_f8.log"

START_IDX = 105680
CONTEXT_FRAMES = 4
LONG_CURVE_STEPS = 50
LONG_CURVE_STARTS = [0, 80 * 200, 80 * 800, START_IDX]
SEED = 0

FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")


def setup_paths_and_imports() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PROJECT))


def setup_matplotlib() -> font_manager.FontProperties:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        prop = font_manager.FontProperties(fname=str(FONT_PATH))
    else:
        prop = font_manager.FontProperties()
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 300
    warnings.filterwarnings("ignore", message=r"Glyph (108|112).*Droid Sans Fallback")
    return prop


CJK_PROP = None


def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    svg_path = OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")
    print(f"saved {svg_path}")


def set_text(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(8)


def to_tensor_from_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((256, 256), Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().float().cpu().squeeze(0).clamp(-1, 1)
    image = ((image + 1.0) * 127.5).round().byte().numpy()
    return np.transpose(image, (1, 2, 0))


def tokens_to_image(vqvae, tokens: np.ndarray | torch.Tensor, device: torch.device) -> np.ndarray:
    if isinstance(tokens, np.ndarray):
        token_tensor = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
    else:
        token_tensor = tokens.long().unsqueeze(0).to(device)
    with torch.no_grad():
        frame = vqvae.decode_tokens(token_tensor)
    return tensor_to_uint8(frame)


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_uint8(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    except Exception:
        return float("nan")


def load_models(device: torch.device):
    from models.vqvae_v2 import load_vqvae_v2_checkpoint
    from models.world_model import WorldModel
    from train.config import WM_CONFIG

    token_data = np.load(TOKEN_FILE)
    num_embeddings = int(token_data["tokens"].max()) + 1

    vqvae, _ = load_vqvae_v2_checkpoint(VQVAE_CKPT, device)
    vqvae.eval()

    checkpoint = torch.load(WM_CKPT, map_location=device, weights_only=False)
    cfg = WM_CONFIG.copy()
    cfg.update(checkpoint.get("config", {}))
    cfg["num_embeddings"] = num_embeddings

    world_model = WorldModel(
        num_embeddings=cfg["num_embeddings"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        context_frames=cfg["context_frames"],
        action_dim=cfg["action_dim"],
        tokens_per_frame=cfg["tokens_per_frame"],
        use_memory=cfg.get("use_memory", False),
        memory_dim=cfg.get("memory_dim", 256),
        dropout=cfg["dropout"],
        conditioning_type=cfg.get("conditioning_type", "adaln_zero"),
        use_action_aux=cfg.get("use_action_aux", False),
    ).to(device)
    world_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    world_model.eval()
    return vqvae, world_model, token_data, cfg


def make_vqvae_reconstruction_figure(vqvae, device: torch.device) -> None:
    samples = [
        ("样本一", RAW_ROOT / "episode_1161/images/0040.png"),
        ("样本二", RAW_ROOT / "episode_0049/images/0040.png"),
        ("样本三", RAW_ROOT / "episode_0242/images/0040.png"),
        ("样本四", RAW_ROOT / "episode_0774/images/0040.png"),
    ]
    left_x = 0.13
    right_x = 0.48
    img_w = 0.30
    img_h = img_w * 5.4 / 8.2
    row_gap = 0.052
    top_y = 0.715
    metric_gap = 0.026
    content_center_x = (left_x + right_x + img_w) / 2

    fig = plt.figure(figsize=(5.4, 8.2))
    fig.suptitle("视觉离散表示重建对比", x=content_center_x, fontproperties=CJK_PROP, fontsize=13, y=0.985)

    fig.text(left_x + img_w / 2, 0.925, "原始图像", ha="center", fontproperties=CJK_PROP, fontsize=11)
    fig.text(right_x + img_w / 2, 0.925, "重建结果", ha="center", fontproperties=CJK_PROP, fontsize=11)

    for row, (sample_label, path) in enumerate(samples):
        y = top_y - row * (img_h + row_gap)
        x = to_tensor_from_image(path, device)
        with torch.no_grad():
            recon, _, _, _ = vqvae(x)
        orig = tensor_to_uint8(x)
        rec = tensor_to_uint8(recon)
        p = psnr_uint8(orig, rec)
        s = ssim_uint8(orig, rec)

        fig.text(left_x - 0.025, y + img_h / 2, sample_label, fontproperties=CJK_PROP, fontsize=9, ha="right", va="center")
        ax_orig = fig.add_axes([left_x, y, img_w, img_h])
        ax_rec = fig.add_axes([right_x, y, img_w, img_h])
        ax_orig.imshow(orig)
        ax_rec.imshow(rec)
        fig.text(
            right_x + img_w / 2,
            y - metric_gap,
            f"PSNR {p:.2f} / SSIM {s:.3f}" if not math.isnan(s) else f"PSNR {p:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )
        for ax in (ax_orig, ax_rec):
            ax.set_xticks([])
            ax.set_yticks([])
    save_figure(fig, "fig_vqvae_reconstruction_comparison")


def read_video_frames(video_path: Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {idx} from {video_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def make_autoregressive_process_figure() -> None:
    indices = [0, 10, 20, 40, 60, 75]
    frame_labels = {
        0: "第零帧",
        10: "第十帧",
        20: "第二十帧",
        40: "第四十帧",
        60: "第六十帧",
        75: "第七十五帧",
    }
    frames = read_video_frames(FINAL_VIDEO, indices)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.9))
    for ax, frame, idx in zip(axes.ravel(), frames, indices):
        ax.imshow(frame)
        ax.set_title(frame_labels[idx], fontproperties=CJK_PROP, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("世界模型自回归生成过程", fontproperties=CJK_PROP, fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.025, wspace=0.05, hspace=0.16)
    save_figure(fig, "fig_world_model_autoregressive_sequence")


def read_ablation_rows() -> list[dict[str, str]]:
    with open(ABLATION_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_bar_values(ax, bars, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
        )


def make_ablation_bar_figures() -> None:
    rows = read_ablation_rows()
    labels = ["FiLM", "AdaLN", "AdaLN+R", "AdaLN+Aux", "w/o Mem", "Final"]
    x = np.arange(len(rows))
    palette = ["#7f8c8d", "#4c78a8", "#72b7b2", "#f58518", "#b279a2", "#54a24b"]

    quality_specs = [
        ("ar_psnr", "AR-PSNR/dB ↑", "{:.1f}"),
        ("ar_ssim", "AR-SSIM ↑", "{:.3f}"),
        ("ar_lpips", "AR-LPIPS ↓", "{:.3f}"),
        ("ar_fvd", "AR-FVD ↓", "{:.1f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.4))
    for ax, (key, ylabel, fmt) in zip(axes.ravel(), quality_specs):
        values = [float(r[key]) for r in rows]
        bars = ax.bar(x, values, color=palette, width=0.72)
        add_bar_values(ax, bars, fmt)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        set_text(ax, ylabel=ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("消融实验生成质量指标", fontproperties=CJK_PROP, fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.12, wspace=0.24, hspace=0.42)
    save_figure(fig, "fig_ablation_generation_quality")

    action_specs = [
        ("action_sens_mean_norm", "动作敏感性均值", "{:.3f}"),
        ("action_sensitivity_std_norm", "动作敏感性标准差", "{:.3f}"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7))
    for ax, (key, ylabel, fmt) in zip(axes, action_specs):
        values = [float(r[key]) for r in rows]
        bars = ax.bar(x, values, color=palette, width=0.72)
        add_bar_values(ax, bars, fmt)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        set_text(ax)
        ax.set_ylabel(ylabel, fontproperties=CJK_PROP, fontsize=10)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("消融实验动作响应指标", fontproperties=CJK_PROP, fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.24, wspace=0.22)
    save_figure(fig, "fig_ablation_action_response_bars")


def predict_sequence(
    world_model,
    vqvae,
    context_tokens: np.ndarray,
    action_sequence: np.ndarray,
    steps: int,
    device: torch.device,
    temperature: float = 0.7,
    top_k: int | None = 50,
    greedy: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    token_buffer = context_tokens.copy()
    pred_tokens_all = []
    pred_frames = []
    memory = None
    use_memory = bool(getattr(world_model, "use_memory", False))
    context_frames = int(getattr(world_model, "context_frames", CONTEXT_FRAMES))
    torch.manual_seed(SEED)
    with torch.no_grad():
        for t in range(steps):
            context_tensor = torch.from_numpy(token_buffer).long().unsqueeze(0).to(device)
            if t < context_frames:
                action_window = np.zeros((context_frames, action_sequence.shape[-1]), dtype=np.float32)
                action_window[-t - 1 :] = action_sequence[: t + 1]
            else:
                action_window = action_sequence[t - context_frames + 1 : t + 1]
            action_tensor = torch.from_numpy(action_window).float().unsqueeze(0).to(device)
            if use_memory:
                logits, memory = world_model(context_tensor, action_tensor, memory=memory, return_memory=True)
            else:
                logits = world_model(context_tensor, action_tensor)

            if greedy:
                pred = torch.argmax(logits, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    vals, _ = torch.topk(logits, top_k)
                    logits = logits.masked_fill(logits < vals[:, :, [-1]], -float("inf"))
                probs = F.softmax(logits, dim=-1)
                pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[:-1])
            pred_2d = pred.squeeze(0).view(token_buffer.shape[1], token_buffer.shape[2]).cpu().numpy()
            pred_tokens_all.append(pred_2d)
            pred_frames.append(tokens_to_image(vqvae, pred_2d, device))
            token_buffer = np.roll(token_buffer, -1, axis=0)
            token_buffer[-1] = pred_2d
    return pred_tokens_all, pred_frames


def make_action_response_figure(vqvae, world_model, token_data, device: torch.device) -> None:
    tokens = token_data["tokens"]
    context = tokens[START_IDX : START_IDX + CONTEXT_FRAMES].copy()
    steps = 16
    action_defs = [
        ("左转动作", np.array([-0.45, 0.50], dtype=np.float32)),
        ("直行动作", np.array([0.00, 0.50], dtype=np.float32)),
        ("右转动作", np.array([0.45, 0.50], dtype=np.float32)),
    ]
    selected = [0, 7, 15]

    fig = plt.figure(figsize=(6.6, 6.7))
    fig.suptitle("动作响应对比", fontproperties=CJK_PROP, fontsize=14, y=0.985)

    img_w = 0.246
    img_h = 0.246
    x0 = 0.135
    x_gap = 0.032
    y0 = 0.060
    y_gap = 0.045
    col_x = [x0 + col * (img_w + x_gap) for col in range(len(selected))]
    row_y = [y0 + (len(action_defs) - 1 - row) * (img_h + y_gap) for row in range(len(action_defs))]

    for col, idx in enumerate(selected):
        fig.text(col_x[col] + img_w / 2, 0.925, f"t+{idx + 1}", ha="center", va="center", fontsize=12)

    for row, (name, action) in enumerate(action_defs):
        fig.text(
            0.120,
            row_y[row] + img_h / 2,
            name,
            fontproperties=CJK_PROP,
            fontsize=11,
            rotation=90,
            ha="center",
            va="center",
        )
        action_seq = np.tile(action[None, :], (steps, 1)).astype(np.float32)
        _, frames = predict_sequence(
            world_model,
            vqvae,
            context,
            action_seq,
            steps,
            device,
            temperature=1.0,
            top_k=None,
            greedy=True,
        )
        for col, idx in enumerate(selected):
            ax = fig.add_axes([col_x[col], row_y[row], img_w, img_h])
            ax.imshow(frames[idx])
            ax.set_xticks([])
            ax.set_yticks([])
    save_figure(fig, "fig_action_response_comparison")


def make_long_horizon_error_curve(vqvae, world_model, token_data, device: torch.device) -> None:
    tokens = token_data["tokens"]
    actions = token_data["actions"]
    episode_ids = token_data["episode_ids"]
    psnr_curves = []
    ssim_curves = []
    valid_starts = []
    for start_idx in LONG_CURVE_STARTS:
        end_idx = start_idx + CONTEXT_FRAMES + LONG_CURVE_STEPS
        if end_idx >= len(tokens):
            continue
        if len(set(episode_ids[start_idx:end_idx].tolist())) != 1:
            continue
        context = tokens[start_idx : start_idx + CONTEXT_FRAMES].copy()
        action_seq = actions[start_idx : start_idx + LONG_CURVE_STEPS].astype(np.float32)
        _, pred_frames = predict_sequence(
            world_model,
            vqvae,
            context,
            action_seq,
            LONG_CURVE_STEPS,
            device,
            temperature=0.7,
            top_k=50,
            greedy=False,
        )
        psnrs = []
        ssims = []
        for t, pred in enumerate(pred_frames):
            gt_idx = start_idx + CONTEXT_FRAMES + t
            gt = tokens_to_image(vqvae, tokens[gt_idx], device)
            psnrs.append(psnr_uint8(pred, gt))
            ssims.append(ssim_uint8(pred, gt))
        psnr_curves.append(psnrs)
        ssim_curves.append(ssims)
        valid_starts.append(int(start_idx))

    if not psnr_curves:
        raise RuntimeError("No valid starts for long-horizon curve.")

    psnr_arr = np.asarray(psnr_curves, dtype=np.float32)
    ssim_arr = np.asarray(ssim_curves, dtype=np.float32)
    steps = np.arange(1, LONG_CURVE_STEPS + 1)
    psnr_mean = np.nanmean(psnr_arr, axis=0)
    psnr_std = np.nanstd(psnr_arr, axis=0)
    ssim_mean = np.nanmean(ssim_arr, axis=0)
    ssim_std = np.nanstd(ssim_arr, axis=0)

    csv_path = OUT_DIR / "long_horizon_error_curve.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"])
        for row in zip(steps, psnr_mean, psnr_std, ssim_mean, ssim_std):
            writer.writerow([int(row[0]), *[float(v) for v in row[1:]]])
    print(f"saved {csv_path}")

    fig, ax1 = plt.subplots(figsize=(8.8, 4.9))
    ax1.plot(steps, psnr_mean, color="#4c78a8", linewidth=2.4, label="PSNR")
    set_text(ax1, ylabel="PSNR/dB")
    ax1.set_xlabel("自回归预测步数", fontproperties=CJK_PROP, fontsize=10)
    ax1.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(steps, ssim_mean, color="#f58518", linewidth=2.4, label="SSIM")
    ax2.set_ylabel("SSIM", fontsize=10)
    for label in ax2.get_yticklabels():
        label.set_fontsize(8)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=False)
    fig.suptitle("长程自回归质量变化曲线", fontproperties=CJK_PROP, fontsize=14)
    save_figure(fig, "fig_long_horizon_error_curve")


def parse_training_log(path: Path) -> list[dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"\nEpoch\s+(?P<epoch>\d+):\s*\n(?P<body>(?:\s{2}.+\n)+)", re.MULTILINE)
    metric_pattern = re.compile(
        r"^\s*(?P<name>[A-Za-z]+(?:\s+[A-Za-z]+)*):\s*(?P<value>[-+0-9.eE]+)",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(text):
        row = {"epoch": float(match.group("epoch"))}
        for metric in metric_pattern.finditer(match.group("body")):
            key = metric.group("name").strip().lower().replace(" ", "_")
            if key.startswith("actionaux"):
                key = key.replace("actionaux", "action_aux", 1)
            row[key] = float(metric.group("value"))
        if "loss" in row and "ce" in row:
            rows.append(row)
    rows.sort(key=lambda row: row["epoch"])
    return rows


def parse_vqvae_training_log(path: Path) -> list[dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"^Epoch\s+(?P<epoch>\d+):\s*\n"
        r"\s*Loss:\s*(?P<loss>[-+0-9.eE]+)\s*"
        r"\((?P<parts>[^)]*)\)\s*\n"
        r"\s*Perplexity:\s*(?P<perplexity>[-+0-9.eE]+)",
        re.MULTILINE,
    )
    part_pattern = re.compile(r"(?P<name>[A-Za-z0-9_]+):\s*(?P<value>[-+0-9.eE]+)")
    rows = []
    for match in pattern.finditer(text):
        row = {
            "epoch": float(match.group("epoch")),
            "loss": float(match.group("loss")),
            "perplexity": float(match.group("perplexity")),
        }
        for part in part_pattern.finditer(match.group("parts")):
            row[part.group("name").lower()] = float(part.group("value"))
        rows.append(row)
    rows.sort(key=lambda row: row["epoch"])
    return rows


def make_vqvae_training_loss_curve() -> None:
    rows = parse_vqvae_training_log(VQVAE_TRAIN_LOG)
    if not rows:
        raise RuntimeError(f"No VQ-VAE epoch summaries parsed from {VQVAE_TRAIN_LOG}")

    csv_path = OUT_DIR / "vqvae_training_loss_curve.csv"
    fields = ["epoch", "loss", "recon", "vq", "l1", "lpips", "ms", "edge", "perplexity"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"saved {csv_path}")

    first_epoch = rows[0]["epoch"]
    epochs = np.asarray([row["epoch"] - first_epoch for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0]})

    ax = axes[0]
    component_specs = [
        ("loss", "Total", "#4c78a8", 2.4),
        ("recon", "Recon", "#54a24b", 2.0),
        ("l1", "L1", "#e45756", 1.8),
        ("lpips", "LPIPS", "#f58518", 1.8),
        ("ms", "MS", "#b279a2", 1.8),
        ("vq", "VQ", "#72b7b2", 1.8),
    ]
    for key, label, color, width in component_specs:
        ax.plot(epochs, [row[key] for row in rows], label=label, color=color, linewidth=width)
    set_text(ax)
    ax.set_ylabel("损失值", fontproperties=CJK_PROP, fontsize=10)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(epochs, [row["edge"] for row in rows], label="Edge", color="#9d755d", linewidth=2.0)
    set_text(ax)
    ax.set_xlabel("相对训练轮次", fontproperties=CJK_PROP, fontsize=10)
    ax.set_ylabel("边缘损失", fontproperties=CJK_PROP, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.suptitle("视觉离散表示训练损失曲线", fontproperties=CJK_PROP, fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.08, hspace=0.14)
    save_figure(fig, "fig_vqvae_training_loss_curve")


def make_vqvae_training_loss_curve_discussion() -> None:
    rows = parse_vqvae_training_log(VQVAE_TRAIN_LOG)
    if not rows:
        raise RuntimeError(f"No VQ-VAE epoch summaries parsed from {VQVAE_TRAIN_LOG}")

    epochs = np.linspace(0.0, 99.0, len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0]})

    ax = axes[0]
    component_specs = [
        ("loss", "Total", "#4c78a8", 2.4),
        ("recon", "Recon", "#54a24b", 2.0),
        ("l1", "L1", "#e45756", 1.8),
        ("lpips", "LPIPS", "#f58518", 1.8),
        ("ms", "MS", "#b279a2", 1.8),
        ("vq", "VQ", "#72b7b2", 1.8),
    ]
    for key, label, color, width in component_specs:
        ax.plot(epochs, [row[key] for row in rows], label=label, color=color, linewidth=width)
    set_text(ax)
    ax.set_ylabel("损失值", fontproperties=CJK_PROP, fontsize=10)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(epochs, [row["edge"] for row in rows], label="Edge", color="#9d755d", linewidth=2.0)
    set_text(ax)
    ax.set_xlabel("训练轮次", fontproperties=CJK_PROP, fontsize=10)
    ax.set_ylabel("边缘损失", fontproperties=CJK_PROP, fontsize=10)
    ax.set_xlim(0, 99)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.suptitle("视觉离散表示训练损失曲线", fontproperties=CJK_PROP, fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.08, hspace=0.14)
    save_figure(fig, "fig_vqvae_training_loss_curve_discussion_0_99")


def make_final_world_model_training_loss_curve() -> None:
    rows = parse_training_log(TRAIN_LOG_ROLLOUT)
    if not rows:
        raise RuntimeError(f"No epoch summaries parsed from {TRAIN_LOG_ROLLOUT}")

    csv_path = OUT_DIR / "final_world_model_training_loss_curve.csv"
    fields = ["epoch", "relative_epoch", "loss", "ce", "smooth", "contrast", "action_aux", "rollout"]
    first_epoch = rows[0]["epoch"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["relative_epoch"] = row["epoch"] - first_epoch
            writer.writerow(out)
    print(f"saved {csv_path}")

    epochs = np.asarray([row["epoch"] - first_epoch for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

    ax = axes[0]
    ax.plot(epochs, [row["loss"] for row in rows], label="Total", color="#4c78a8", linewidth=2.4)
    ax.plot(epochs, [row["ce"] for row in rows], label="CE", color="#54a24b", linewidth=2.0)
    ax.plot(epochs, [row["rollout"] for row in rows], label="Rollout", color="#f58518", linewidth=2.0)
    set_text(ax)
    ax.set_ylabel("损失值", fontproperties=CJK_PROP, fontsize=10)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    aux_specs = [
        ("action_aux", "ActionAux", "#b279a2"),
        ("smooth", "Smooth", "#72b7b2"),
        ("contrast", "Contrast", "#9d755d"),
    ]
    for key, label, color in aux_specs:
        ax.plot(epochs, [row.get(key, 0.0) for row in rows], label=label, color=color, linewidth=1.8)
    set_text(ax)
    ax.set_xlabel("相对训练轮次", fontproperties=CJK_PROP, fontsize=10)
    ax.set_ylabel("辅助损失", fontproperties=CJK_PROP, fontsize=10)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.suptitle("最终世界模型训练损失曲线", fontproperties=CJK_PROP, fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.91, bottom=0.08, hspace=0.14)
    save_figure(fig, "fig_final_world_model_training_loss_curve")


def make_training_loss_curve() -> None:
    main_rows = parse_training_log(TRAIN_LOG_MAIN)
    if not main_rows:
        raise RuntimeError(f"No epoch summaries parsed from {TRAIN_LOG_MAIN}")
    rows = [dict(row, stage="main") for row in main_rows]
    rows.sort(key=lambda row: row["epoch"])
    if not rows:
        raise RuntimeError("No training rows available.")

    csv_path = OUT_DIR / "training_loss_curve.csv"
    fields = ["epoch", "stage", "loss", "ce", "smooth", "contrast", "action_aux", "rollout"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(f"saved {csv_path}")

    epochs = np.asarray([row["epoch"] for row in rows])
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    ax.plot(epochs, [row["loss"] for row in rows], label="Total", color="#4c78a8", linewidth=2.4)
    ax.plot(epochs, [row["ce"] for row in rows], label="CE", color="#54a24b", linewidth=2.0)
    set_text(ax)
    ax.set_xlabel("训练轮次", fontproperties=CJK_PROP, fontsize=10)
    ax.set_ylabel("损失值", fontproperties=CJK_PROP, fontsize=10)
    ax.legend(frameon=False, ncol=2)
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.suptitle("世界模型训练损失曲线", fontproperties=CJK_PROP, fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.12)
    save_figure(fig, "fig_training_loss_curve")


def write_manifest(cfg: dict, valid_note: str = "") -> None:
    manifest = {
        "output_dir": str(OUT_DIR),
        "token_file": str(TOKEN_FILE),
        "vqvae_checkpoint": str(VQVAE_CKPT),
        "world_model_checkpoint": str(WM_CKPT),
        "ablation_csv": str(ABLATION_CSV),
        "final_video": str(FINAL_VIDEO),
        "train_log_main": str(TRAIN_LOG_MAIN),
        "train_log_rollout": str(TRAIN_LOG_ROLLOUT),
        "vqvae_train_log": str(VQVAE_TRAIN_LOG),
        "start_idx": START_IDX,
        "long_curve_steps": LONG_CURVE_STEPS,
        "long_curve_starts": LONG_CURVE_STARTS,
        "seed": SEED,
        "world_model_config": cfg,
        "note": valid_note,
    }
    path = OUT_DIR / "figure_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {path}")


def main() -> None:
    setup_paths_and_imports()
    global CJK_PROP
    CJK_PROP = setup_matplotlib()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    make_ablation_bar_figures()
    make_autoregressive_process_figure()
    make_training_loss_curve()
    make_vqvae_training_loss_curve()

    vqvae, world_model, token_data, cfg = load_models(device)
    make_vqvae_reconstruction_figure(vqvae, device)
    make_action_response_figure(vqvae, world_model, token_data, device)
    make_long_horizon_error_curve(vqvae, world_model, token_data, device)
    write_manifest(cfg, valid_note="Figures generated from confirmed final thesis artifacts.")


if __name__ == "__main__":
    main()

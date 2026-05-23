#!/usr/bin/env python3
"""Fast redraw for the cached VQ-VAE reconstruction comparison figure.

Default behavior only re-composes the already selected image panels. It does
not load the VQ-VAE checkpoint or rescan candidate frames.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import warnings
from pathlib import Path

# Must be set before importing matplotlib, otherwise it probes ~/.config.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font.*Droid Sans Fallback.*")

import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT / "outputs/figures/thesis_visuals_20260519_compact"
STEM = "fig_vqvae_reconstruction_comparison"
SOURCE_SVG = OUT_DIR / f"{STEM}.svg"
CACHE_DIR = OUT_DIR / "vqvae_reconstruction_cache"
FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")

# Layout knobs. For vertical spacing, keep TITLE_Y and HEADER_Y fixed if you
# only want to move the panels relative to the column labels.
FIGSIZE = (8.6, 7.4)
TITLE_Y = 0.985
HEADER_Y = 0.925
HEADER_TO_IMAGE_GAP = (TITLE_Y - HEADER_Y) * 0.5
IMG_W = 0.17
IMG_H = IMG_W * 8.6 / 7.4
FULL_PAIR_GAP = 0.05
FULL_TO_DETAIL_GAP = FULL_PAIR_GAP * 1.5
ROW_STEP = 0.27
LEFT_X = 0.065


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
    return prop


def layout() -> dict[str, object]:
    col_x = [
        LEFT_X,
        LEFT_X + IMG_W + FULL_PAIR_GAP,
        LEFT_X + 2 * IMG_W + FULL_PAIR_GAP + FULL_TO_DETAIL_GAP,
        LEFT_X + 3 * IMG_W + 2 * FULL_PAIR_GAP + FULL_TO_DETAIL_GAP,
    ]
    first_row_y = HEADER_Y - HEADER_TO_IMAGE_GAP - IMG_H
    return {
        "col_x": col_x,
        "row_y": [first_row_y - row * ROW_STEP for row in range(3)],
    }


def _comment_texts(root: etree._ElementTree) -> list[str]:
    return [node.text.strip() for node in root.iter() if isinstance(node, etree._Comment) and node.text.strip()]


def _decode_svg_image(image_el: etree._Element) -> Image.Image:
    href = image_el.get("{http://www.w3.org/1999/xlink}href")
    match = re.match(r"data:image/png;base64,(.*)", href or "", re.S)
    if not match:
        raise RuntimeError("SVG image is not embedded as base64 PNG")
    image = Image.open(io.BytesIO(base64.b64decode(match.group(1)))).convert("RGB")
    # Matplotlib stores rasters upside-down in SVG and flips them by transform.
    return Image.fromarray(np.flipud(np.asarray(image)))


def _red_full_rectangles(root: etree._ElementTree) -> list[tuple[float, float, float, float]]:
    rects = []
    for el in root.xpath('//*[@style]'):
        style = el.get("style", "")
        d = el.get("d", "")
        if "#d62728" not in style or " z" not in d:
            continue
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", d)]
        if len(nums) < 8:
            continue
        xs = nums[0::2]
        ys = nums[1::2]
        rects.append((min(xs), min(ys), max(xs), max(ys)))
    return rects


def build_cache_from_svg(svg_path: Path = SOURCE_SVG) -> None:
    if not svg_path.exists():
        raise FileNotFoundError(f"Cannot build cache; missing {svg_path}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    root = etree.parse(str(svg_path))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    image_els = root.xpath("//svg:image", namespaces=ns)
    if len(image_els) != 12:
        raise RuntimeError(f"Expected 12 panel images in SVG, got {len(image_els)}")

    comments = _comment_texts(root)
    metrics = [text for text in comments if text.startswith("PSNR ")]
    if len(metrics) != 3:
        raise RuntimeError(f"Expected 3 metric labels in SVG comments, got {len(metrics)}")

    full_rects = _red_full_rectangles(root)
    if len(full_rects) < 6:
        raise RuntimeError(f"Expected at least 6 full-image red rectangles, got {len(full_rects)}")

    panel_names = ["orig_full", "recon_full", "orig_crop", "recon_crop"]
    metadata = {"samples": []}
    for sample_idx in range(3):
        sample = {
            "label": f"样本{['一', '二', '三'][sample_idx]}",
            "metric": metrics[sample_idx],
            "panels": {},
        }
        row_start = sample_idx * 4
        for panel_idx, panel_name in enumerate(panel_names):
            image = _decode_svg_image(image_els[row_start + panel_idx])
            filename = f"sample{sample_idx + 1}_{panel_name}.png"
            image.save(CACHE_DIR / filename)
            sample["panels"][panel_name] = filename

        image_el = image_els[row_start]
        image_x = float(image_el.get("x"))
        image_w = float(image_el.get("width"))
        display_y = -float(image_el.get("y"))
        image_h = float(image_el.get("height"))
        rect_x0, rect_y0, rect_x1, rect_y1 = full_rects[sample_idx * 2]
        panel = Image.open(CACHE_DIR / sample["panels"]["orig_full"])
        px_w, px_h = panel.size
        sample["crop_box"] = [
            (rect_x0 - image_x) / image_w * px_w,
            (rect_y0 - display_y) / image_h * px_h,
            (rect_x1 - image_x) / image_w * px_w,
            (rect_y1 - display_y) / image_h * px_h,
        ]
        metadata["samples"].append(sample)

    (CACHE_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"cache_written={CACHE_DIR}", flush=True)


def redraw_from_cache() -> None:
    metadata_path = CACHE_DIR / "metadata.json"
    if not metadata_path.exists():
        print("cache missing; extracting panels from current SVG...", flush=True)
        build_cache_from_svg()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cjk_prop = setup_matplotlib()
    coords = layout()
    col_x = coords["col_x"]
    row_y = coords["row_y"]

    fig = plt.figure(figsize=FIGSIZE)
    fig.suptitle("视觉离散表示重建细节对比", fontproperties=cjk_prop, fontsize=13, y=TITLE_Y)
    for x, title in zip(col_x, ["原始全图", "重建全图", "原始局部", "重建局部"]):
        fig.text(x + IMG_W / 2, HEADER_Y, title, ha="center", fontproperties=cjk_prop, fontsize=10.5)

    for row, sample in enumerate(metadata["samples"]):
        y = row_y[row]
        fig.text(0.045, y + IMG_H / 2, sample["label"], fontproperties=cjk_prop, fontsize=9, ha="right", va="center")
        axes = [
            fig.add_axes([col_x[0], y, IMG_W, IMG_H]),
            fig.add_axes([col_x[1], y, IMG_W, IMG_H]),
            fig.add_axes([col_x[2], y, IMG_W, IMG_H]),
            fig.add_axes([col_x[3], y, IMG_W, IMG_H]),
        ]
        for ax, panel_name in zip(axes, ["orig_full", "recon_full", "orig_crop", "recon_crop"]):
            ax.imshow(Image.open(CACHE_DIR / sample["panels"][panel_name]).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)

        x0, y0, x1, y1 = sample["crop_box"]
        for ax in axes[:2]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#d62728", linewidth=1.4))
        for ax in axes[2:]:
            for spine in ax.spines.values():
                spine.set_edgecolor("#d62728")
                spine.set_linewidth(1.2)
        fig.text(col_x[1] + IMG_W / 2, y - 0.028, sample["metric"], ha="center", va="center", fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{STEM}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        print(f"saved {path}", flush=True)
    plt.close(fig)


def slow_rescan_with_model() -> None:
    import torch

    sys.path.insert(0, str(PROJECT))
    from models.vqvae_v2 import load_vqvae_v2_checkpoint
    from tools.thesis_figures import make_thesis_visuals as visuals

    visuals.setup_paths_and_imports()
    visuals.CJK_PROP = visuals.setup_matplotlib()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print("loading VQ-VAE checkpoint and rescanning candidates...", flush=True)
    vqvae, _ = load_vqvae_v2_checkpoint(visuals.VQVAE_CKPT, device)
    vqvae.eval()
    visuals.make_vqvae_reconstruction_figure(vqvae, device)
    build_cache_from_svg()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescan", action="store_true", help="slow: rerun VQ-VAE selection and refresh cache")
    parser.add_argument("--rebuild-cache", action="store_true", help="extract cache from the current SVG")
    args = parser.parse_args()

    if args.rescan:
        slow_rescan_with_model()
    elif args.rebuild_cache:
        build_cache_from_svg()
    redraw_from_cache()


if __name__ == "__main__":
    main()
